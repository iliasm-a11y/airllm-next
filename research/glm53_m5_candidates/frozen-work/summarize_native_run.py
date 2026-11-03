"""Summarize recorded native evidence without promoting diagnostics to EOS."""
import argparse
from collections import Counter
import json
from pathlib import Path

def summarize(root):
    guard = json.loads((root / 'guard.json').read_text())
    path = root / 'capture/report.json'
    captured = json.loads(path.read_text()) if path.exists() else {}
    samples = guard.get('samples', [])
    result = {
        'schema_version': 1, 'source_run': str(root.resolve()),
        'guard_status': guard['status'], 'native_returncode': guard.get('returncode'),
        'guard_violation': guard.get('guard_violation'),
        'power_context_passed': guard.get('power_context_passed'),
        'integrity_passed': guard.get('integrity_passed'),
        'elapsed_seconds': guard.get('elapsed_seconds'),
        'microbatch': guard['microbatch'], 'mode': guard['mode'],
        'capture_complete': captured.get('complete', False),
        'diagnostic': captured.get('diagnostic'),
        'functional_eos_qualified': False,
        'full_snapshot_parity_qualified': False,
        'qualification_scope': 'descriptive evidence only; separate strict comparisons and useful natural EOS are required',
        'sampled_peak_rss_bytes': max((s.get('rss_bytes', 0) for s in samples), default=None),
        'observed_peak_swap_growth_bytes': max((s.get('swap_growth_bytes', 0) for s in samples), default=None),
        'sampled_minimum_available_bytes': min((s['available_bytes'] for s in samples), default=None),
        'pressure_level_counts': dict(Counter(s['pressure_level'] for s in samples)),
        'samples': len(samples), 'model_loads': captured.get('model_loads'),
        'model_mapping_setup_seconds': captured.get('model_load_seconds'),
        'physical_slots': captured.get('physical_slots'),
        'snapshot': captured.get('snapshot'), 'requests': [],
        'snapshot_key_sha256': captured.get('snapshot_key_sha256'),
        'source_sha256': guard.get('preflight', {}).get('source_sha256'),
        'runtime_binaries': guard.get('preflight', {}).get('runtime_binaries'),
        'unobserved': {'pure_weight_wait_seconds': None, 'pure_cpu_math_seconds': None,
            'gpu_compute_seconds': None, 'synchronization_seconds': None,
            'physical_ssd_read_bytes': None, 'two_useful_natural_eos_seconds': None},
    }
    for request in captured.get('requests', []):
        row = dict(request)
        before, after = row.pop('before', {}), row.pop('after', {})
        row['process_counter_deltas'] = {key: after[key] - before[key]
            for key in ('cpu_seconds', 'os_storage_read_bytes', 'pageins')
            if key in before and key in after}
        row['counter_scope'] = 'process user+system CPU and OS-accounted reads, not pure math or physical SSD transfers'
        row['functional_eos_qualified'] = False
        result['requests'].append(row)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.run)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('guard_status', 'capture_complete', 'microbatch', 'mode', 'functional_eos_qualified')}))
