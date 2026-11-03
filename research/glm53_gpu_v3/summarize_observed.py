#!/usr/bin/env python3
"""Summarize closed native campaigns offline; never samples hardware or runs a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


REQUEST_SCOPE = 'whole request, including prefill and decode; endpoint snapshots are not atomic'
BLOCK_SCOPE = ('sum of observed system-wide IOBlockStorageDriver counters; includes other processes '
               'and may include multiple device layers; not per-PID transfers or NAND bytes')
GPU_SCOPE = ('request-sliced owned completed Metal command buffers, including copies/events; '
             'capture completeness is not guaranteed; not per-kernel compute time')


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _counter(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _delta(before, after, key):
    a, b = (before or {}).get(key), (after or {}).get(key)
    if not _counter(a) or not _counter(b):
        return {'value': None, 'reason': f'{key}: missing or invalid endpoint'}
    if b < a:
        return {'value': None, 'reason': f'{key}: counter reset'}
    return {'value': b-a, 'reason': None}


def _timebase(report):
    """Use measured host timebase associated with this owned PID, never a fixed machine ratio."""
    ratios, identities = set(), set()
    pid = report.get('owned_pid')
    for sample in report.get('samples', []):
        owned = sample.get('owned_process') or {}
        if not owned.get('available') or owned.get('pid') != pid:
            continue
        identity = owned.get('identity') or {}
        ratio = owned.get('mach_timebase') or {}
        numer, denom = ratio.get('numer'), ratio.get('denom')
        if identity.get('pid') != pid or not _counter(identity.get('start_abstime')):
            return {'value': None, 'reason': 'owned PID identity missing'}
        identities.add((pid, identity['start_abstime']))
        if not _counter(numer) or not _counter(denom) or not numer or not denom:
            return {'value': None, 'reason': 'invalid measured Mach timebase'}
        ratios.add((numer, denom))
    if len(ratios) != 1 or len(identities) != 1:
        return {'value': None, 'reason': 'missing or inconsistent owned PID/timebase snapshots'}
    numer, denom = next(iter(ratios))
    return {'value': {'numer': numer, 'denom': denom},
            'owned_identity': {'pid': pid, 'start_abstime': next(iter(identities))[1]},
            'reason': None}


def _block_reads(request):
    before, after = request.get('block_device_before') or {}, request.get('block_device_after') or {}
    if not before.get('available') or not after.get('available'):
        return {'value': None, 'reason': 'missing device endpoint snapshot', 'scope': BLOCK_SCOPE}
    old_rows, new_rows = before.get('devices', []), after.get('devices', [])
    old = {row.get('registry_id'): row for row in old_rows}
    new = {row.get('registry_id'): row for row in new_rows}
    if (not old or None in old or set(old) != set(new) or
            len(old) != len(old_rows) or len(new) != len(new_rows)):
        return {'value': None, 'reason': 'device set changed or ambiguous identity', 'scope': BLOCK_SCOPE}
    rows = []
    for identity in sorted(old):
        delta = _delta(old[identity].get('statistics'), new[identity].get('statistics'), 'Bytes (Read)')
        if delta['value'] is None:
            return {'value': None, 'reason': delta['reason'], 'scope': BLOCK_SCOPE}
        rows.append({'registry_id': identity, 'bytes': delta['value']})
    return {'value': sum(row['bytes'] for row in rows), 'devices': rows, 'reason': None,
            'scope': BLOCK_SCOPE}


def summarize_request(job, stage, report, request, index):
    before, after = request.get('usage_before'), request.get('usage_after')
    ticks = {key: _delta(before, after, key) for key in ('user_ticks', 'system_ticks')}
    timebase = _timebase(report)
    ratio = timebase['value']
    cpu_seconds = None
    if ratio and all(item['value'] is not None for item in ticks.values()):
        cpu_seconds = sum(item['value'] for item in ticks.values()) * ratio['numer']/ratio['denom']/1e9
    final = request.get('final') or {}
    gpu = request.get('gpu_command_times') or {}
    gpu_fields = {key: gpu.get(key) for key in ('recorded_commands', 'valid_commands',
        'sum_command_seconds', 'union_command_seconds', 'complete_capture_guaranteed')}
    return {
        'job_index': job.get('index'), 'cycle': job.get('cycle'), 'arm': job.get('arm'),
        'candidate': job.get('candidate'), 'stage_status': stage.get('status'),
        'report_status': report.get('status'), 'request_index': index,
        'guard_violation': report.get('guard_violation'), 'report_error': report.get('error'),
        'observed_decode_tokens_per_second': request.get('observed_decode_tokens_per_second'),
        'decode_rate_scope': 'first through last token-bearing SSE event, after first token; runner-derived',
        'server_timings': final.get('timings'),
        'ttft_seconds': request.get('ttft_seconds'), 'request_wall_seconds': request.get('wall_seconds'),
        'load_seconds': report.get('load_seconds'), 'total_seconds': report.get('total_seconds'),
        'peak_rss_bytes': report.get('peak_rss_bytes'), 'peak_footprint_bytes': report.get('peak_footprint_bytes'),
        'peak_system_swap_growth_bytes': report.get('peak_swap_growth_bytes'),
        'request_owned_pid_os_read_bytes': _delta(before, after, 'storage_read_bytes'),
        'request_owned_pid_pageins': _delta(before, after, 'pageins'),
        'request_global_block_read_bytes': _block_reads(request),
        'request_cpu_ticks': ticks, 'recorded_mach_timebase': timebase,
        'request_owned_pid_cpu_seconds': cpu_seconds,
        'cpu_seconds_scope': 'sum of owned process user+system CPU time; can exceed request wall time',
        'request_gpu_command_times': {**gpu_fields, 'scope': GPU_SCOPE},
        'gpu_commands_required': job.get('gpu_commands_required'),
        'counter_scope': REQUEST_SCOPE,
        'stop_type': final.get('stop_type'), 'terminal_stop_observed': final.get('stop') is True,
        'generated_token_ids': request.get('tokens'),
        'generated_token_count': len(request['tokens']) if isinstance(request.get('tokens'), list) else None,
        'requested_tokens': (request.get('body') or {}).get('n_predict'),
        'prefill_tokens_evaluated': final.get('tokens_evaluated'),
        'decode_only_counter_delta': None,
        'decode_counter_unavailable_reason': ('request usage endpoints span prefill+decode; sampled snapshots '
            'do not identify exact decode boundaries and no interpolation is justified'),
    }


def summarize_campaign(path):
    path = Path(path)
    if path.is_dir():
        path = path/'campaign.json'
    encoded = path.read_bytes()
    campaign = json.loads(encoded)
    stages, plan = campaign.get('stages', []), campaign.get('plan') or {}
    if campaign.get('status') not in ('complete', 'failed', 'cancelled'):
        raise ValueError('Only a closed complete/failed/cancelled campaign may be summarized')
    if any(stage.get('status') in ('running', 'planned', None) for stage in stages):
        raise ValueError('Campaign contains a stage without a final status')
    if not isinstance(plan.get('jobs'), list):
        raise ValueError('Campaign plan has no jobs list')
    rows, reports = [], []
    for position, job in enumerate(plan['jobs']):
        stage = stages[position] if position < len(stages) else {'status': 'not_launched'}
        report_path = Path(job['report'])
        if not report_path.is_absolute():
            report_path = path.parent/report_path
        report = stage.get('report_data') or {}
        record = {'job_index': job.get('index'), 'path': str(report_path), 'exists': report_path.is_file()}
        if report_path.is_file():
            raw = report_path.read_bytes()
            disk_report = json.loads(raw)
            record.update(sha256=hashlib.sha256(raw).hexdigest(),
                          matches_embedded_report=(disk_report == report) if report else None)
            report = disk_report
        reports.append(record)
        requests = report.get('requests') or []
        if requests:
            rows.extend(summarize_request(job, stage, report, request, i) for i, request in enumerate(requests))
        else:
            rows.append(summarize_request(job, stage, report, {}, None))
    if path.read_bytes() != encoded:
        raise ValueError('Campaign manifest changed while summarizing')
    return {
        'schema': 1, 'campaign_path': str(path.resolve()),
        'campaign_sha256': hashlib.sha256(encoded).hexdigest(),
        'summarizer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'campaign_status': campaign['status'], 'recorded_campaign_summary': campaign.get('summary'),
        'rows': rows, 'raw_report_sources': reports,
        'optimization_qualified': False,
        'qualification_scope': ('descriptive offline transcription, not independent provenance/fidelity '
            'validation, causal attribution, or acceptance of a speed gain'),
        'unknowns': ['physical NAND bytes', 'per-PID physical device transfers', 'cache-hit bytes',
            'decode-only I/O/CPU/GPU deltas', 'isolated dequantization/router/allocation/copy/wait durations'],
        'limits': ['OS file cache uncontrolled; fresh process is not proof of cold storage',
            'counter endpoints are sequential and span different small instrumentation intervals',
            'system swap/controller counters include unrelated background processes',
            'RSS and physical footprint measure different things; one is not a decomposition of the other',
            'no per-token storage conversion, service-time wall fraction, or NAND/cache-hit inference'],
    }


def markdown(summary):
    def fmt(value):
        if value is None:
            return 'unknown'
        if isinstance(value, float):
            return f'{value:.6f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')

    headings = ['Run', 'Stage/report', 'tok/s', 'TTFT s', 'Request s', 'Load s', 'Total s',
        'Peak RSS B', 'Peak system swap growth B', 'PID OS read B*', 'Global block read B*',
        'PID CPU s*', 'Mach numer/denom', 'GPU commands*/union s*', 'Stop', 'Tokens']
    lines = [f"Closed campaign: `{summary['campaign_path']}`", '',
        '* counters cover the entire request, including prefill. Block reads are system-wide; '
        'GPU intervals include copies/events. CPU seconds sum process threads.', '',
        '| ' + ' | '.join(headings) + ' |', '| ' + ' | '.join(['---']*len(headings)) + ' |']
    for row in summary['rows']:
        ratio = row['recorded_mach_timebase']['value']
        gpu = row['request_gpu_command_times']
        values = [f"{row['job_index']}-{row['arm']}/req{row['request_index']}",
            f"{row['stage_status']}/{row['report_status']}", row['observed_decode_tokens_per_second'],
            row['ttft_seconds'], row['request_wall_seconds'], row['load_seconds'], row['total_seconds'],
            row['peak_rss_bytes'], row['peak_system_swap_growth_bytes'],
            row['request_owned_pid_os_read_bytes']['value'], row['request_global_block_read_bytes']['value'],
            row['request_owned_pid_cpu_seconds'], f"{ratio['numer']}/{ratio['denom']}" if ratio else None,
            f"{fmt(gpu['valid_commands'])}/{fmt(gpu['union_command_seconds'])}",
            row['stop_type'], f"{fmt(row['generated_token_count'])}/{fmt(row['requested_tokens'])}"]
        lines.append('| ' + ' | '.join(fmt(value) for value in values) + ' |')
    lines.extend(['', 'Descriptive table only; no optimization is qualified by this helper.', '',
        'Decode-only counter deltas, cache-hit bytes, per-PID physical transfers and NAND bytes remain unknown.'])
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args(argv)
    destinations = [args.output] + ([args.markdown] if args.markdown else [])
    if len({str(path.resolve()) for path in destinations}) != len(destinations):
        parser.error('JSON and Markdown outputs must be different files')
    if any(path.exists() for path in destinations):
        parser.error('Refusing to overwrite summary evidence')
    result = summarize_campaign(args.campaign)
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')
    if args.markdown:
        with args.markdown.open('x') as out:
            out.write(markdown(result))


if __name__ == '__main__':
    main()
