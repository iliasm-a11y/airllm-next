#!/usr/bin/env python3
"""Independently check eight HyperStream workers using only the standard library.

No engine/analyzer imports, model reads, inference, or GPU access. Recorded
before/after manifests certify the measurement interval. --check-current-inputs
additionally hashes current small engine/protocol files; it never opens weights.
The delivered decode rate is recomputed from total_seconds minus ttft_seconds,
not copied from the runtime throughput field or the primary analyzer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


class PendingEvidence(ValueError):
    pass


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, positive=False):
    demand(type(value) in (int, float) and math.isfinite(value), 'Nonfinite/invalid ' + name)
    demand(value > 0 if positive else value >= 0, 'Negative/zero ' + name)
    return value


def integer(value, name):
    demand(type(value) is int and value >= 0, 'Invalid integer ' + name)
    return value


def digest_map(value, name):
    demand(isinstance(value, dict) and value, 'Missing ' + name)
    demand(all(isinstance(k, str) and re.fullmatch(r'[0-9a-f]{64}', v or '')
               for k, v in value.items()), 'Invalid digest map ' + name)


def argument(command, flag):
    demand(isinstance(command, list) and command.count(flag) == 1, 'Missing/duplicate ' + flag)
    position = command.index(flag)
    demand(position + 1 < len(command), 'Missing value for ' + flag)
    return command[position + 1]


def verify(folder, summary=None, check_current=False):
    inputs = []
    def read(path):
        raw = path.read_bytes()
        inputs.append({'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    campaign = read(folder / 'campaign.json')
    if campaign.get('status') != 'completed':
        raise PendingEvidence('Campaign is not completed: ' + str(campaign.get('status')))
    demand(campaign.get('protocol_unchanged') is True and campaign.get('engines_unchanged') is True,
           'Campaign inputs were not preserved')
    manifests = campaign.get('engine_manifests')
    demand(isinstance(manifests, dict) and len(manifests) == 2, 'Expected two frozen engine roots')
    demand(campaign.get('engine_manifests_after') == manifests, 'Engine manifests changed')
    protocol = campaign.get('protocol_sha256')
    digest_map(protocol, 'protocol')
    current_hash_count = 0
    if check_current:
        for path, expected in protocol.items():
            demand(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected,
                   'Current protocol differs: ' + path)
            current_hash_count += 1
    for root, manifest in manifests.items():
        demand(re.fullmatch(r'[0-9a-f]{40}', manifest.get('git_head', '')), 'Invalid engine revision')
        digest_map(manifest.get('source_sha256'), 'engine sources')
        if check_current:
            current = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(root) / 'air_llm/airllm/nextgen').glob('*.py')}
            demand(current == manifest['source_sha256'], 'Current engine sources differ: ' + root)
            current_hash_count += len(current)
    jobs, runs = campaign.get('plan'), campaign.get('runs')
    demand(isinstance(jobs, list) and isinstance(runs, list) and len(jobs) == len(runs) == 8,
           'Expected exactly eight planned and completed workers')
    demand([j.get('id') for j in jobs] == [r.get('id') for r in runs], 'Run order differs from plan')
    demand(len({j['id'] for j in jobs}) == 8, 'Repeated worker label')
    pairs, rows, common_identity, common_limits = {}, [], None, None
    for job, run in zip(jobs, runs):
        label = job['id']
        match = re.fullmatch(r'(short|explanation)\.r([01])\.(baseline|candidate)', label)
        demand(match is not None, 'Unexpected worker label: ' + label)
        task, repeat, variant = match.group(1), int(match.group(2)), match.group(3)
        demand(run.get('status') == 'completed' and type(run.get('exit_code')) is int
               and run['exit_code'] == 0, 'Incomplete/failed run: ' + label)
        worker = read(folder / (label + '.json'))
        guard = read(folder / (label + '.guard.json'))
        command = run.get('command')
        demand(guard.get('command') == command + ['--worker']
               and worker.get('command') == command[1:] + ['--worker'], 'Command chain mismatch: ' + label)
        demand(Path(argument(command, '--output')).name == label + '.json', 'Output command mismatch')
        demand(Path(run.get('guard_file', '')).name == label + '.guard.json', 'Guard filename mismatch')
        for flag, expected in (('--engine-root', job['engine_root']), ('--tasks', job['tasks']), ('--label', label)):
            demand(argument(command, flag) == expected, 'Plan command mismatch: ' + flag)
        demand(command[-len(job['args']):] == job['args'], 'Comparison controls differ from plan')
        manifest = manifests[job['engine_root']]
        demand(json.loads(argument(command, '--expected-engine-sources-json')) == manifest['source_sha256']
               and argument(command, '--expected-engine-head') == manifest['git_head']
               and json.loads(argument(command, '--expected-protocol-json')) == protocol,
               'Expected source/protocol command mismatch')
        demand(worker.get('status') == 'completed' and worker.get('label') == label, 'Incomplete worker')
        demand(worker.get('engine_root') == job['engine_root']
               and worker.get('source_sha256') == manifest['source_sha256']
               and worker.get('git_head') == worker.get('git_head_after') == manifest['git_head'],
               'Worker source/revision mismatch')
        for field in ('source_unchanged', 'protocol_unchanged', 'model_stat_unchanged', 'engine_head_unchanged'):
            demand(worker.get(field) is True, 'Worker identity changed: ' + field)
        tasks = campaign['tasks_by_job'][label]
        demand(len(tasks) == 1 and worker.get('tasks_snapshot') == tasks, 'Task snapshot mismatch')
        demand(worker.get('protocol_sha256') == protocol
               and worker.get('harness_sha256') == protocol[command[1]]
               and worker.get('tasks_sha256') == protocol[job['tasks']], 'Protocol digest mismatch')
        demand(type(guard.get('exit_code')) is int and guard['exit_code'] == 0
               and 'violation' in guard and guard['violation'] is None
               and not guard.get('exception') and not guard.get('cleanup_error'), 'Guard failed')
        limits = guard.get('limits')
        expected_limits = {'rss_bytes': 2 << 30, 'reserve_bytes': 3 << 30,
                           'swap_growth_bytes': 128 << 20, 'admitted_payload_bytes': 768 << 20,
                           'timeout_seconds': 600., 'interval_seconds': .05, 'max_processes': 32}
        demand(limits == expected_limits, 'Unexpected or relaxed guard limits')
        if common_limits is None:
            common_limits = limits
        demand(limits == common_limits, 'Limits differ between workers')
        demand(integer(guard.get('sample_count'), 'samples') > 0, 'Guard did not sample')
        demand(integer(guard.get('sampled_peak_rss_bytes'), 'peak RSS') <= limits['rss_bytes'], 'RSS limit crossed')
        demand(integer(guard.get('minimum_available_bytes'), 'minimum available') >= limits['reserve_bytes'],
               'System reserve crossed')
        demand(integer(guard.get('maximum_swap_growth_bytes'), 'swap growth') <= limits['swap_growth_bytes'],
               'Swap growth limit crossed')
        demand(guard.get('peak_pressure_level') == 1, 'Elevated/unknown memory pressure')
        demand(integer(guard.get('sampled_peak_processes'), 'process count') <= limits['max_processes'],
               'Process limit crossed')
        for owner in (guard, worker):
            for side in ('before', 'after'):
                snap = owner.get(side, {})
                for field in ('physical_bytes', 'available_bytes', 'swap_used_bytes'):
                    integer(snap.get(field), 'snapshot ' + field)
                demand(snap.get('pressure_level') == 1, 'Snapshot pressure is elevated or missing')
        demand(guard['before']['available_bytes'] >= limits['reserve_bytes'] + limits['admitted_payload_bytes'],
               'Guard admission had insufficient reserve plus payload')
        if run.get('launch_wait_status') is not None:
            demand(run['launch_wait_status'] == 'ready', 'Resource window was not ready')
            policy = run.get('launch_wait_policy', {})
            demand(policy.get('admission_limits') == limits and policy.get('required_pressure_level') == 1,
                   'Resource-window policy relaxed limits')
            samples = run.get('launch_wait_samples', [])
            demand(samples and samples[-1].get('ready') is True
                   and samples[-1].get('observed_stable_seconds', -1) >= 5., 'Resource window lacks stability evidence')
        identity = {key: worker.get(key) for key in ('model', 'hardware', 'versions', 'cache_regime')}
        if common_identity is None:
            common_identity = identity
        demand(identity == common_identity, 'Model/machine/runtime/cache identity differs')
        model = worker.get('model', {})
        demand(integer(model.get('bytes'), 'model bytes') > worker['before']['physical_bytes'],
               'Checkpoint is not larger than recorded physical memory')
        demand(argument(command, '--model') == model.get('path'), 'Model command/stat path mismatch')
        demand(worker.get('physical_ssd_bandwidth_certified') is False, 'Unexpected physical-SSD claim')
        options, backend = worker.get('options', {}), worker.get('backend_options', {})
        demand(options.get('reserve_bytes') == 3 << 30 and options.get('dtype') == 'float32'
               and options.get('device') == 'gpu' and options.get('prefetch') is False,
               'Unexpected precision/device/reserve/prefetch')
        turns = worker.get('turns')
        demand(isinstance(turns, list) and len(turns) == 1, 'Expected one fresh-worker turn')
        turn = turns[0]
        demand(all(turn.get(k) == tasks[0].get(k) for k in ('id', 'prompt', 'max_new_tokens')),
               'Turn differs from frozen task')
        demand(turn.get('id') == task and turn.get('status') == 'completed'
               and turn.get('stop_reason') == 'eos', 'Turn incomplete or truncated')
        generated, stops, supplied = turn.get('generated_ids'), turn.get('eos_token_ids'), turn.get('full_input_ids')
        demand(isinstance(generated, list) and 2 <= len(generated) <= tasks[0]['max_new_tokens']
               and isinstance(stops, list) and stops and isinstance(supplied, list) and supplied, 'Invalid token arrays')
        for token in generated + stops + supplied:
            integer(token, 'token identity')
        demand(generated[-1] in stops and not any(t in stops for t in generated[:-1]), 'Invalid terminal EOS')
        demand(turn.get('partial_ids') == generated and turn.get('full_output_ids') == supplied + generated
               and turn.get('appended_input_ids') == supplied and turn.get('reused_prefix_tokens') == 0,
               'Fresh-worker/emitted sequence mismatch')
        params = turn.get('generation_parameters', {})
        demand(params.get('temperature') == 0 and params.get('do_sample') is False
               and params.get('max_new_tokens') == tasks[0]['max_new_tokens'], 'Unexpected generation controls')
        demand(turn.get('finite_logits_checked') is True, 'Finite logits were not checked')
        steps, timing = turn.get('step_checks'), turn.get('token_seconds')
        demand(isinstance(steps, list) and len(steps) == len(generated)
               and isinstance(timing, list) and len(timing) == len(generated), 'Missing token checks/times')
        for index, step in enumerate(steps):
            demand(step.get('position') == len(supplied) + index, 'Causal position mismatch')
            for field in ('logits_sha256', 'routes_sha256'):
                demand(re.fullmatch(r'[0-9a-f]{64}', step.get(field, '')), 'Malformed logits/routes digest')
            integer(step.get('logical_bytes'), 'step logical bytes')
            if step.get('process_storage_read_bytes') is not None:
                integer(step['process_storage_read_bytes'], 'step OS storage bytes')
            number(step.get('forward_seconds'), 'forward seconds', positive=True)
        demand(turn.get('native_cache_position') == steps[-1]['position'], 'Final cache position mismatch')
        demand(sum(s['logical_bytes'] for s in steps) == turn.get('logical_read_bytes'), 'Logical bytes do not reconcile')
        for seconds in timing:
            number(seconds, 'token seconds', positive=True)
        total = number(turn.get('total_seconds'), 'generation seconds', positive=True)
        ttft = number(turn.get('ttft_seconds'), 'first-token seconds', positive=True)
        elapsed = number(guard.get('elapsed_seconds'), 'guard elapsed', positive=True)
        demand(total > ttft and elapsed >= worker.get('worker_wall_seconds', math.inf), 'Invalid timing ordering')
        decode_seconds = total - ttft
        rate = (len(generated) - 1) / decode_seconds
        demand(math.isclose(rate, turn.get('decode_tokens_per_second', -1), rel_tol=1e-8, abs_tol=1e-8),
               'Runtime decode field disagrees with independent total-minus-first calculation')
        metrics = turn.get('metrics', {})
        mlx_peak = metrics.get('allocator', {}).get('peak_bytes')
        if mlx_peak is not None:
            integer(mlx_peak, 'MLX peak')
        os_bytes = turn.get('process_storage_read_bytes')
        if os_bytes is not None:
            integer(os_bytes, 'OS storage bytes')
        row = {'label': label, 'task': task, 'repeat': repeat, 'variant': variant,
               'generated_tokens': len(generated), 'decode_tokens': len(generated) - 1,
               'decode_seconds': decode_seconds, 'decode_tok_s': rate, 'generation_seconds': total,
               'guard_elapsed_seconds': elapsed, 'ttft_seconds': ttft,
               'end_to_end_tok_s': len(generated) / elapsed,
               'rss_peak_bytes': guard['sampled_peak_rss_bytes'], 'mlx_peak_bytes': mlx_peak,
               'minimum_available_bytes': guard['minimum_available_bytes'],
               'swap_initial_bytes': guard['before']['swap_used_bytes'],
               'swap_final_bytes': guard['after']['swap_used_bytes'],
               'swap_growth_bytes': guard['maximum_swap_growth_bytes'],
               'logical_read_bytes': turn['logical_read_bytes'], 'process_storage_read_bytes': os_bytes,
               'decode_logical_bytes': sum(s['logical_bytes'] for s in steps[1:]),
               'decode_storage_bytes': None if any(s['process_storage_read_bytes'] is None for s in steps[1:])
                   else sum(s['process_storage_read_bytes'] for s in steps[1:])}
        rows.append(row)
        pairs.setdefault((task, repeat), {})[variant] = (worker, turn, row)
    demand(len(pairs) == 4 and all(set(p) == {'baseline', 'candidate'} for p in pairs.values()), 'Missing exact pairs')
    paired = []
    for (task, repeat), variants in sorted(pairs.items()):
        bw, bt, br = variants['baseline']
        cw, ct, cr = variants['candidate']
        demand(bw['options'] == cw['options'], 'Paired model options differ')
        bbackend, cbackend = dict(bw['backend_options']), dict(cw['backend_options'])
        demand(cbackend.pop('expert_pipeline', False) is True
               and cbackend.pop('string_buffer_bytes', 0) == 65536, 'Candidate HyperStream controls missing')
        demand(bbackend.pop('expert_pipeline', False) is False
               and bbackend.pop('string_buffer_bytes', 0) == 0 and bbackend == cbackend,
               'Uncontrolled backend differences')
        for field in ('full_input_ids', 'appended_input_ids', 'generated_ids', 'full_output_ids', 'eos_token_ids',
                      'generation_parameters', 'native_cache_position', 'text', 'raw_text'):
            demand(bt.get(field) == ct.get(field), 'Paired difference: ' + field)
        for field in ('logits_sha256', 'routes_sha256', 'position'):
            demand([s[field] for s in bt['step_checks']] == [s[field] for s in ct['step_checks']],
                   'Paired per-token difference: ' + field)
        paired.append({'task': task, 'repeat': repeat, 'tokens_logits_routes_positions_text_exact': True,
                       'decode_speedup': cr['decode_tok_s'] / br['decode_tok_s'],
                       'end_to_end_speedup': cr['end_to_end_tok_s'] / br['end_to_end_tok_s']})
    for task in ('short', 'explanation'):
        orders = [[r['variant'] for r in rows if r['task'] == task and r['repeat'] == repeat] for repeat in (0, 1)]
        demand(orders[0] == list(reversed(orders[1])), 'Pair order is not balanced across repetitions')
    aggregate = {}
    for variant in ('baseline', 'candidate'):
        selected = [r for r in rows if r['variant'] == variant]
        tokens = sum(r['generated_tokens'] for r in selected)
        decoded = sum(r['decode_tokens'] for r in selected)
        def optional_sum(field):
            return None if any(r[field] is None for r in selected) else sum(r[field] for r in selected)
        aggregate[variant] = {
            'workers': len(selected), 'generated_tokens_including_eos': tokens,
            'decode_tokens_including_final_eos': decoded,
            'stream_decode_seconds': sum(r['decode_seconds'] for r in selected),
            'decode_tok_s': decoded / sum(r['decode_seconds'] for r in selected),
            'end_to_end_tok_s': tokens / sum(r['guard_elapsed_seconds'] for r in selected),
            'guard_elapsed_seconds': sum(r['guard_elapsed_seconds'] for r in selected),
            'peak_owned_rss_bytes_sampled': max(r['rss_peak_bytes'] for r in selected),
            'peak_mlx_allocator_bytes': None if any(r['mlx_peak_bytes'] is None for r in selected)
                else max(r['mlx_peak_bytes'] for r in selected),
            'minimum_available_bytes_sampled': min(r['minimum_available_bytes'] for r in selected),
            'maximum_system_swap_growth_bytes_sampled': max(r['swap_growth_bytes'] for r in selected),
            'logical_read_bytes': sum(r['logical_read_bytes'] for r in selected),
            'process_storage_read_bytes': optional_sum('process_storage_read_bytes'),
            'decode_logical_read_bytes_per_token': sum(r['decode_logical_bytes'] for r in selected) / decoded,
            'decode_process_storage_read_bytes_per_token': None if optional_sum('decode_storage_bytes') is None
                else optional_sum('decode_storage_bytes') / decoded,
            'median_ttft_seconds': statistics.median(r['ttft_seconds'] for r in selected),
        }
    result = {'status': 'verified', 'worker_count': 8, 'pair_count': 4,
              'all_tokens_logits_routes_positions_text_exact': True,
              'aggregate': aggregate, 'pairs': paired, 'workers': rows,
              'aggregate_decode_speedup': aggregate['candidate']['decode_tok_s'] / aggregate['baseline']['decode_tok_s'],
              'aggregate_end_to_end_speedup': aggregate['candidate']['end_to_end_tok_s'] / aggregate['baseline']['end_to_end_tok_s'],
              'current_small_input_hashes_checked': current_hash_count,
              'model_identity': common_identity['model'],
              'model_integrity_scope': 'Recorded immutable stat identity; no model read or full-file SHA256 recomputation',
              'memory_scope': 'Sampled owned-process RSS and overlapping MLX allocator peak are separate measures',
              'storage_scope': 'Logical requests and process OS-accounted reads are separate; no cold-SSD/NAND certification',
              'swap_scope': 'Initial and final nonzero system swap retained per worker; growth is not total usage',
              'inputs': inputs}
    if summary is not None:
        primary = read(summary)
        demand(primary.get('worker_count') == 8 and primary.get('pair_count') == 4, 'Primary summary cardinality differs')
        comparisons = 0
        for variant in ('baseline', 'candidate'):
            for field, value in aggregate[variant].items():
                if field == 'median_ttft_seconds':
                    other = primary['aggregate'][variant]['ttft_seconds']['median']
                else:
                    other = primary['aggregate'][variant][field]
                demand(value is None and other is None or type(value) in (int, float)
                       and type(other) in (int, float) and math.isclose(value, other, rel_tol=1e-10, abs_tol=1e-8),
                       'Primary aggregate mismatch: ' + variant + '/' + field)
                comparisons += 1
        for field in ('aggregate_decode_speedup', 'aggregate_end_to_end_speedup'):
            demand(math.isclose(result[field], primary[field], rel_tol=1e-10, abs_tol=1e-10), 'Primary speedup mismatch')
            comparisons += 1
        result['primary_summary_numeric_comparisons'] = comparisons
    for item in inputs:
        demand(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'],
               'Evidence changed during independent verification: ' + item['path'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-current-inputs', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = verify(args.campaign.resolve(), args.summary, args.check_current_inputs)
        code = 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {'status': 'pending' if isinstance(error, PendingEvidence) else 'rejected',
                  'error': f'{type(error).__name__}: {error}'}
        code = 2
    result['checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('inputs', 'workers', 'aggregate', 'pairs')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
