#!/usr/bin/env python3
"""Read-only forensic comparison without weakening the retained fidelity gate.

This reports ordered expert differences separately from selected-set differences
and checks every retained emitted logit. Router scores and hidden states absent
from the original recording cannot be reconstructed or declared equivalent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from verify_gguf_speculation import _recording, _row_values, digest_file


def diagnose(serial_report, candidate_report, atol=1e-4, rtol=1e-4):
    serial = _recording(serial_report, serial=True)
    candidate = _recording(candidate_report, serial=False)
    if serial['model'] != candidate['model'] or serial['config'] != candidate['config']:
        raise ValueError('Checkpoint/config identity differs; comparisons are not admissible')
    if len(serial['turns']) != len(candidate['turns']):
        raise ValueError('Turn counts differ')
    output = dict(schema='airllm-m5-pair-forensics-v1', status='diagnostic_completed',
        scope='Recorded accepted-prefix ordered routes, selected expert sets, emitted full logits and tokens only',
        serial_report_sha256=serial['sha256'], candidate_report_sha256=candidate['sha256'],
        diagnostic_source_sha256=digest_file(Path(__file__)), atol=atol,rtol=rtol,
        router_scores_recorded=False,router_inputs_recorded=False,
        hidden_state_parity_validated=False, performance_gain_qualified=False, turns=[])
    for a,b in zip(serial['turns'],candidate['turns']):
        identity_equal = all(a[k] == b[k] for k in ('id','prefix','generated','eos','stop'))
        if not identity_equal:
            raise ValueError('Turn transcript/EOS differs; this diagnostic requires matching consumed prefixes')
        differences=[]
        for key in sorted(set(a['routes'])|set(b['routes']),key=lambda k:(k[1],int(k[0]))):
            left,right=a['routes'].get(key),b['routes'].get(key)
            if left != right:
                differences.append(dict(layer=int(key[0]),position=key[1],serial=left,candidate=right,
                    selected_set_equal=left is not None and right is not None and set(left)==set(right)))
        rows=[]
        for output_index,reference_rows in a['rows'].items():
            reference=_row_values(reference_rows[0])
            for row_ref in b['rows'][output_index]:
                actual=_row_values(row_ref)
                maximum=relative=0.
                scaled=0.
                violating=0
                bitwise=True
                worst={}
                for token,(expected,observed) in enumerate(zip(reference,actual)):
                    error=abs(observed-expected)
                    allowed=atol+rtol*abs(expected)
                    maximum=max(maximum,error)
                    relative=max(relative,error/max(abs(expected),1e-30))
                    violating+=error>allowed
                    if error/allowed>scaled:
                        scaled=error/allowed
                        worst=dict(vocabulary_index=token,serial=expected,candidate=observed,
                                   absolute_error=error,allowed_error=allowed)
                    bitwise=bitwise and struct.pack('<f',expected)==struct.pack('<f',observed)
                rows.append(dict(output_index=output_index,elements=len(reference),
                    max_absolute_error=maximum,max_relative_error=relative,
                    violating_elements=violating,bitwise_equal=bitwise,
                    worst_tolerance_scaled_error=scaled,worst_tolerance_scaled_element=worst))
        output['turns'].append(dict(id=a['id'],generated_ids=a['generated'],eos=a['eos'],stop=a['stop'],
            accepted_prefix_route_positions=len(a['routes']),ordered_route_differences=differences,
            ordered_route_difference_count=len(differences),
            selected_set_difference_count=sum(not x['selected_set_equal'] for x in differences),
            logits=rows,full_logits_elements_checked=sum(row['elements'] for row in rows),
            full_logits_within_retained_tolerance=all(row['violating_elements']==0 for row in rows),
            strict_ordered_route_gate_passed=not differences))
    for recording in (serial,candidate):
        if digest_file(recording['path']) != recording['sha256']:
            raise ValueError('Recording changed during diagnosis')
        for path,expected in recording['artifacts'].items():
            if digest_file(path) != expected:
                raise ValueError('Logits artifact changed during diagnosis')
    output['strict_existing_fidelity_gate_passed']=all(
        turn['strict_ordered_route_gate_passed'] and turn['full_logits_within_retained_tolerance']
        for turn in output['turns'])
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('serial_report',type=Path)
    p.add_argument('candidate_report',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    result=diagnose(a.serial_report,a.candidate_report)
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'],strict_existing_fidelity_gate_passed=
                         result['strict_existing_fidelity_gate_passed'],output=str(a.output))))


if __name__=='__main__':
    main()
