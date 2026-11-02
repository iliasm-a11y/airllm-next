"""Device-global block IO and owned Metal command timestamps, with scope limits."""
import plistlib
from pathlib import Path
import re
import subprocess


def block_snapshot():
    try:
        raw = subprocess.check_output(['/usr/sbin/ioreg', '-r', '-c', 'IOBlockStorageDriver', '-a'], timeout=5)
        rows = plistlib.loads(raw)
        return {'available': True, 'devices': [
            {'registry_id': row.get('IORegistryEntryID'), 'name': row.get('IORegistryEntryName'),
             'statistics': row.get('Statistics', {})} for row in rows],
            'scope': 'system-wide IOBlockStorageDriver counters; controller block reads, not per-process IO or NAND cells'}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {'available': False, 'error': str(exc)}


def compare_snapshots(before, after):
    if not before.get('available') or not after.get('available'):
        return {'available': False}
    keys = ('Bytes (Read)', 'Bytes (Write)', 'Operations (Read)', 'Operations (Write)',
            'Total Time (Read)', 'Total Time (Write)')
    old = {r['registry_id']: r for r in before['devices']}
    current = {r['registry_id']: r for r in after['devices']}
    if set(old) != set(current) or None in old or len(old) != len(before['devices']) or len(current) != len(after['devices']):
        return {'available': False, 'error': 'device set changed or ambiguous device identity'}
    values = []
    for row in after['devices']:
        if row['registry_id'] not in old:
            continue
        a, b = old[row['registry_id']]['statistics'], row['statistics']
        delta = {k: b[k]-a[k] for k in keys if k in a and k in b}
        if any(v < 0 for v in delta.values()):
            return {'available': False, 'error': 'device counter reset'}
        values.append({'registry_id': row['registry_id'], 'delta': delta})
    return {'available': bool(values), 'devices': values,
            'scope': before['scope'],
            'limits': 'Includes other processes. Summed service time can overlap and is not a wall-time IO fraction.'}


def parse_gpu_times(log, start_command=0):
    text = Path(log).read_text(errors='replace')
    pattern = r'AIRLLM_GPU_TIME origin=(\S+) start=([\d.]+) end=([\d.]+) seconds=([-\d.]+) status=(\d+)'
    rows = [{'origin': a, 'start': float(b), 'end': float(c), 'seconds': float(d), 'status': int(e)}
            for a, b, c, d, e in re.findall(pattern, text)]
    rows = rows[start_command:]
    valid = [r for r in rows if r['end'] >= r['start'] > 0 and r['status'] == 4 and
             r['seconds'] >= 0 and abs((r['end']-r['start'])-r['seconds']) <= 5e-8 + r['seconds']*1e-6]
    intervals = sorted((r['start'], r['end']) for r in valid)
    union = 0.0
    end = None
    for left, right in intervals:
        union += right-left if end is None else max(0, right-max(end, left))
        end = right if end is None else max(end, right)
    return {'recorded_commands': len(rows), 'valid_commands': len(valid),
            'sum_command_seconds': sum(r['seconds'] for r in valid), 'union_command_seconds': union,
            'rows': rows, 'scope': 'owned completed Metal command buffers including copies/events; not per-kernel counters',
            'complete_capture_guaranteed': False}
