#!/usr/bin/env python3
"""Compare saved CPU-prefix tensors on identical forced token trajectories."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def compare_prefix(reference: Path, candidate: Path, first_gpu_layer: int = 40):
    source = Path(__file__).resolve().parents[1] / 'glm53_m5/compare_fidelity.py'
    spec = importlib.util.spec_from_file_location('pr17_fidelity', source)
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    a, b = comparator.read_capture(reference), comparator.read_capture(candidate)
    if a['prompt'] != b['prompt'] or [x['next_input_token'] for x in a['tokens']] != [x['next_input_token'] for x in b['tokens']]:
        raise ValueError('CPU-prefix comparison requires identical input trajectories')

    def select(capture):
        return {key: row for key, row in capture['tensors'].items()
                if (match := re.search(r'-(\d+)', row['name'])) and int(match[1]) < first_gpu_layer}

    aa, bb = select(a), select(b)
    rows = []
    hashes = [hashlib.sha256(), hashlib.sha256()]
    for key in sorted(set(aa) | set(bb)):
        step, name, occurrence = key
        row = {'step': step, 'name': name, 'occurrence': occurrence,
               'family': comparator.family(name), 'comparable': True}
        if key not in aa or key not in bb:
            rows.append({**row, 'passed': False, 'error': 'Missing tensor on one side'})
            continue
        views = []
        for i, (root, tensor) in enumerate(((reference, aa[key]), (candidate, bb[key]))):
            views.append(comparator.tensor_view(root, tensor))
            hashes[i].update(json.dumps(key).encode())
            hashes[i].update(comparator.safe_file(root, tensor['file']).read_bytes())
        rows.append({**row, **comparator.compare_arrays(*views, comparator.CRITERIA['cpu'], integer=aa[key]['family'] == 'routes')})
    coverage = {family: comparator.aggregate([row for row in rows if row['family'] == family])
                for family in ('routes', 'router', 'kda')}
    sources = []
    for root, capture, digest in ((reference, a, hashes[0]), (candidate, b, hashes[1])):
        sources.append({'capture': root.name, 'run': capture['run'],
                        'manifest_sha256': hashlib.sha256((root / 'tensors.jsonl').read_bytes()).hexdigest(),
                        'selected_payload_sequence_sha256': digest.hexdigest(),
                        'environment': json.loads((root / 'environment.json').read_text()) if (root / 'environment.json').is_file() else None})
    return {'schema_version': 1, 'scope': 'Saved CPU-prefix diagnostic tensors including captured aliases; no throughput claim',
            'layers': [0, first_gpu_layer - 1], 'sources': sources, 'prompt_identical': True,
            'forced_inputs_identical': True, 'coverage': coverage,
            'tensor_manifest_keys_equal_in_prefix': set(aa) == set(bb),
            'tensor_count': len(rows), 'total_elements': sum(row.get('elements', 0) for row in rows),
            'first_differences': sorted((row for row in rows if not row.get('passed', False)),
                                        key=lambda row: (row['step'], int(re.search(r'-(\d+)', row['name'])[1])))[:10],
            'passed': bool(rows) and all(row.get('passed', False) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--first-gpu-layer', type=int, choices=range(1, 46), default=40)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare_prefix(args.reference, args.candidate, args.first_gpu_layer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'tensor_count': result['tensor_count'],
                      'total_elements': result['total_elements']}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
