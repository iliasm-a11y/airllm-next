#!/usr/bin/env python3
"""Summarize one completed guarded placement diagnostic, never throughput."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from research.glm53_m5 import compare_fidelity as fidelity
from .audit_evidence import first
from .audit_hc_ablation import compare_prefix


def summarize(run, reference):
    candidate = run / 'capture'
    comparison = json.loads((run / 'comparison.json').read_text())
    guard = json.loads((run / 'guard.json').read_text())
    experiment = json.loads((run / 'experiment.json').read_text())
    a, b = fidelity.read_capture(reference), fidelity.read_capture(candidate)
    if a['prompt'] != b['prompt'] or [x['next_input_token'] for x in a['tokens']] != [x['next_input_token'] for x in b['tokens']]:
        raise ValueError('Retest audit requires identical forced trajectories')
    first_observed = {}
    for family in ('kda', 'router', 'routes'):
        rows = [row for row in comparison['tensors'] if row['family'] == family
                and row.get('comparable') and 'reshaped' not in row['name']]
        first_observed[family] = {
            'bit_difference': first(rows, lambda row: row.get('bit_mismatch_count', 0) > 0),
            'criterion_failure': first(rows, lambda row: row.get('mismatch_count', 0) > 0)}
    changes, vectors = [], 0
    for key, row in a['tensors'].items():
        if row['family'] != 'routes' or 'reshaped' in row['name']:
            continue
        if key not in b['tensors']:
            raise ValueError('Retest is missing a canonical route tensor')
        x = fidelity.tensor_view(reference, row).reshape(8, -1)
        y = fidelity.tensor_view(candidate, b['tensors'][key]).reshape(8, -1)
        if x.shape != y.shape:
            raise ValueError('Retest route shapes disagree')
        for column in range(x.shape[1]):
            vectors += 1
            differences = int(np.count_nonzero(x[:, column] != y[:, column]))
            if differences:
                changes.append({'step': row['step'], 'layer': int(re.search(r'-(\d+)', row['name'])[1]),
                                'token_column': column, 'different_positions': differences,
                                'membership_changed': set(x[:, column]) != set(y[:, column]),
                                'reference': x[:, column].tolist(), 'candidate': y[:, column].tolist()})
    log = (run / 'guard.stderr.log').read_text()
    cache = re.findall(r'ubatch <= 8: hits = (\d+), misses = (\d+), hit rate = ([\d.]+)%, uploaded = ([\d.]+) MiB', log)
    cache_stats = None
    if cache:
        hits, misses, rate, uploaded = cache[-1]
        cache_stats = {'hits': int(hits), 'misses': int(misses), 'reported_hit_rate_percent': float(rate),
                       'reported_uploaded_mib': float(uploaded), 'scope': 'native cache counters, not SSD physical traffic'}
    return {'schema_version': 1, 'run_name': run.name,
            'scope': 'Forced-input tensor diagnostic; serialized captures; elapsed time is not decode throughput',
            'source_sha256': {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                              for name in ('comparison.json', 'experiment.json', 'guard.json', 'guard.stderr.log')},
            'environment': experiment['environment'], 'probe_source_sha256': experiment['probe_source_sha256'],
            'binary_sha256': guard['binary_sha256'], 'reference_run': a['run'], 'candidate_run': b['run'],
            'passed': comparison['passed'], 'criteria': comparison['criteria'],
            'full_logits_summary': comparison['full_logits_summary'], 'coverage': comparison['coverage'],
            'first_observed': first_observed,
            'routes': {'vectors': vectors, 'ordered_different': len(changes),
                       'membership_different': sum(int(row['membership_changed']) for row in changes),
                       'different_positions': sum(row['different_positions'] for row in changes),
                       'changes': sorted(changes, key=lambda row: (row['step'], row['layer'], row['token_column']))},
            'cpu_prefix': compare_prefix(reference, candidate), 'cache_stats': cache_stats,
            'guard': {key: guard[key] for key in ('status', 'returncode', 'guard_violation', 'elapsed_seconds',
                      'peak_sampled_rss_bytes', 'peak_system_swap_growth_bytes', 'last_sampled_os_storage_read_bytes',
                      'last_sampled_cpu_seconds', 'measurement_scope')},
            'manifest_errors': comparison['errors'], 'new_decode_gain_accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'fidelity_passed': report['passed'], 'routes': report['routes']['ordered_different'],
                      'cpu_prefix_bitexact': report['cpu_prefix']['passed']}))


if __name__ == '__main__':
    main()
