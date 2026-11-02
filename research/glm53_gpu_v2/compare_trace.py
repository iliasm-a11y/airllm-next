"""Compare named layer-40 outputs in actual graph evaluation order."""
import argparse
from collections import Counter
import json
from pathlib import Path
from research.glm53_m5.compare_fidelity import tensor_view, compare_arrays, CRITERIA


def compare(reference, candidate):
    def rows(root):
        seen = Counter()
        result = []
        for line in (root / 'tensors.jsonl').read_text().splitlines():
            row = json.loads(line)
            k = (row['step'], row['name'])
            result.append(((*k, seen[k]), row))
            seen[k] += 1
        return result
    aa, bb = rows(reference), dict(rows(candidate))
    details = []
    for key, a in aa:
        if '-40' not in key[1] and key[1] != 'result_norm':
            continue
        if key not in bb:
            details.append({'key': key, 'missing_candidate': True})
            continue
        b = bb[key]
        metrics = compare_arrays(tensor_view(reference, a), tensor_view(candidate, b),
                                 CRITERIA['metal'], integer=a['type'] == 26)
        details.append({'key': key, 'op': a.get('op'), 'shape': a['shape'], **metrics})
    return {'schema_version': 1, 'scope': 'named diagnostic outputs, in CPU evaluation order; not every internal operation',
            'first_bit_difference': next((r for r in details if r.get('bit_mismatch_count', 0)), None),
            'first_tolerance_failure': next((r for r in details if not r.get('passed', False)), None),
            'rows': details}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    result = compare(a.reference, a.candidate)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
