"""Recompute inherited decode diagnostics; keep them separate from current runs."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
from .profile_analysis import conditional_bound

def analyze(repository):
    root=Path(repository)
    summary=root/'research/glm53_gpu_v2/evidence/cpu-profile16-summary.json'
    compact=json.loads(summary.read_text())
    raw=gzip.decompress((summary.parent/compact['full_op_layer_step_summary']).read_bytes())
    if hashlib.sha256(raw).hexdigest()!=compact['uncompressed_full_summary_sha256']:
        raise ValueError('historical operation summary hash differs')
    full=json.loads(raw);ops={};layers={}
    for key,value in full['by_op_layer_step'].items():
        op,layer,step=key.split('/')
        if int(step.split('=')[1])==0:continue
        for groups,label in ((ops,op),(layers,layer)):
            row=groups.setdefault(label,{'wall_seconds':0.,'cpu_core_seconds':0.,'os_read_bytes':0,'operations':0})
            row['wall_seconds']+=value['op_wall_seconds']
            row['cpu_core_seconds']+=value['cpu_seconds']
            row['os_read_bytes']+=value['os_read_bytes']
            row['operations']+=value['operations']
    total=compact['by_phase']['decode'];intervals=len(compact['by_step'])-1
    for row in ops.values():row['fraction_of_serialized_operation_wall']=row['wall_seconds']/total['op_wall_seconds']
    union_path=root/'research/glm53_gpu_v3/evidence/cpu-stream16-source-union.json'
    union=json.loads(union_path.read_text())
    decoded=[row for row in union['per_step'] if row['step']>0]
    if not union['decode_steps_closed']:raise ValueError('incomplete historical logical-byte capture')
    logical=statistics.median(row['distinct_known_matmul_operand_bytes'] for row in decoded)
    dense=statistics.median(row['distinct_dense_operand_bytes'] for row in decoded)
    moe=statistics.median(row['distinct_selected_expert_operand_bytes'] for row in decoded)
    return {'schema':1,'scope':'reanalysis of PR19 operation diagnostics and separate PR21 logical-weight captures; no contemporary model run',
        'sources':{str(summary.relative_to(root)):hashlib.sha256(summary.read_bytes()).hexdigest(),
                   str(union_path.relative_to(root)):hashlib.sha256(union_path.read_bytes()).hexdigest(),
                   'decompressed_operation_summary_sha256':hashlib.sha256(raw).hexdigest()},
        'historical_decode_intervals':intervals,'serialized_decode_operation_wall_seconds':total['op_wall_seconds'],
        'historical_process_os_read_bytes':total['os_read_bytes'],
        'historical_process_os_read_bytes_per_interval':total['os_read_bytes']/intervals,
        'historical_cpu_core_seconds':total['cpu_seconds'],'by_operation':ops,'by_layer':layers,
        'logical_operands_separate_capture':{'median_bytes_per_token':logical,'dense_bytes':dense,
            'selected_moe_bytes':moe,'dense_fraction':dense/logical,'moe_fraction':moe/logical,
            'scope':'distinct original operands; not measured SSD/NAND traffic'},
        'conditional_bounds':[conditional_bound(logical,coverage,bandwidth,compute,overlap)
            for coverage in (0,4e9,8e9) for bandwidth in (1e9,2e9,3e9)
            for compute in (0,1) for overlap in (True,False)],
        'target_required_coverage_before_compute':[{'tokens_per_second':target,'assumed_interface_bytes_per_second':bandwidth,
            'minimum_coverage_bytes':max(0,logical-bandwidth/target)}
            for target in (.5,1,3,5) for bandwidth in (1e9,2e9,3e9)],
        'unknowns':{'physical_nand_bytes':None,'pure_io_wait_seconds':None,'unavoidable_compute_seconds':None,
                    'allocation_bytes':None,'contemporary_model_decode_rate':None},
        'interpretation':['CPU seconds sum active threads and cannot be subtracted from wall time to obtain IO wait',
            'operation callbacks serialize execution; these operation sums are not production token throughput',
            'separate historical versions/captures are not an ABBA comparison',
            'conditional coverage includes every cache capable of supplying original bytes; it is not the RSS',
            'bandwidth1/2/3GB/s, coverage0/4/8GB and compute0/1s are explicit hypotheses only']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=analyze(Path(__file__).resolve().parents[2])
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'historical_decode_intervals':report['historical_decode_intervals'],
        'logical':report['logical_operands_separate_capture'],
        'dense_wall_fraction':report['by_operation']['MUL_MAT']['fraction_of_serialized_operation_wall']}))
