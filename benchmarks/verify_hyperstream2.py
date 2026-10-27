#!/usr/bin/env python3
"""Recalculate guarded HyperStream 2 pairs without engine, NumPy or analyzer imports.

The existing campaign_research51 schema is retained. A separate, predeclared
campaign-integrity.json identifies the expected changed controls, numerical
tolerances and before/after checkpoint hashes. Float32 artifacts are checked
directly with hashlib and struct; recorded pair-validation results are unused.
This checker verifies recorded evidence, not authenticity or physical SSD I/O.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct


class RejectedEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise RejectedEvidence(message)


def numeric(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite or invalid ' + name)
    require(value > 0 if positive else value >= 0, 'Negative or zero ' + name)
    return value


def natural(value, name):
    require(type(value) is int and value >= 0, 'Invalid integer ' + name)
    return value


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid digest ' + name)
    return value


def digest_map(value, name):
    require(isinstance(value, dict) and value, 'Missing digest map ' + name)
    for key, item in value.items():
        require(isinstance(key, str), 'Invalid digest-map key')
        digest(item, name + '/' + key)


def argument(command, flag):
    require(isinstance(command, list) and command.count(flag) == 1, 'Missing or repeated ' + flag)
    index = command.index(flag)
    require(index + 1 < len(command), 'Missing value for ' + flag)
    return command[index + 1]


def pair_identity(job):
    label = job.get('id', '')
    match = re.fullmatch(r'(baseline|candidate)-p(\d+)', label)
    if match:
        return int(match[2]), match[1]
    match = re.fullmatch(r'(.+)\.r(\d+)\.(baseline|candidate)', label)
    require(match is not None, 'Unsupported pair label: ' + str(label))
    return (match[1], int(match[2])), match[3]


def describe(values):
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                maximum=max(values), mean=statistics.fmean(values))


def verify(folder, integrity=None, *, check_current=False):
    folder = Path(folder).resolve()
    evidence_hashes = {}

    def read_bytes(path):
        path = Path(path)
        raw = path.read_bytes()
        value = hashlib.sha256(raw).hexdigest()
        require(path not in evidence_hashes or evidence_hashes[path] == value, 'Evidence changed: ' + str(path))
        evidence_hashes[path] = value
        return raw

    def read_json(path):
        return json.loads(read_bytes(path), parse_constant=lambda value: (_ for _ in ()).throw(RejectedEvidence(value)))

    campaign = read_json(folder / 'campaign.json')
    integrity_path = Path(integrity) if integrity is not None else folder / 'campaign-integrity.json'
    attestation = read_json(integrity_path)
    require(campaign.get('status') == 'completed', 'Campaign did not complete')
    require(campaign.get('protocol_unchanged') is True and campaign.get('engines_unchanged') is True,
            'Campaign inputs drifted')
    manifests = campaign.get('engine_manifests')
    require(isinstance(manifests, dict) and manifests, 'No frozen engine manifests')
    require(campaign.get('engine_manifests_after') == manifests, 'Engine manifests changed')
    protocol = campaign.get('protocol_sha256')
    digest_map(protocol, 'protocol')
    for root, manifest in manifests.items():
        require(isinstance(root, str) and re.fullmatch('[0-9a-f]{40}', manifest.get('git_head', '')),
                'Invalid engine revision')
        digest_map(manifest.get('source_sha256'), 'engine sources')
        if check_current:
            current = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(root) / 'air_llm/airllm/nextgen').glob('*.py')}
            require(current == manifest['source_sha256'], 'Current engine sources differ: ' + root)
    if check_current:
        for path, expected in protocol.items():
            require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, 'Current protocol changed: ' + path)

    model_integrity = attestation.get('model_integrity', {})
    before_hash = digest(model_integrity.get('before_sha256'), 'checkpoint before')
    require(digest(model_integrity.get('after_sha256'), 'checkpoint after') == before_hash,
            'Checkpoint hashes differ')
    require(model_integrity.get('stable_file') is True, 'Checkpoint identity was not stable')
    model_bytes = natural(model_integrity.get('bytes'), 'checkpoint size')
    require(model_bytes > 0, 'Empty checkpoint')
    minimum_pairs = natural(attestation.get('minimum_pairs'), 'minimum pairs')
    require(minimum_pairs > 0, 'Minimum pair count must be positive')
    comparison = attestation.get('comparison', {})
    numerical = comparison.get('numerical', {})
    mode = numerical.get('mode')
    require(mode in ('exact', 'tolerance'), 'Numerical validation mode missing')
    rtol, atol = numerical.get('rtol', 0.), numerical.get('atol', 0.)
    for name, value in (('rtol', rtol), ('atol', atol)):
        numeric(value, name)
        require(value <= 1e-4, 'Tolerance exceeds reviewed 1e-4 bound')
    require(mode != 'exact' or (rtol == 0 and atol == 0), 'Exact validation has nonzero tolerances')
    allowed = {}
    for field, valid_keys in (('options', {'prefill_chunk'}),
                              ('backend_options', {'hyper_prefill', 'hyper_prefill_max_tokens'})):
        changes = comparison.get(field, {})
        require(isinstance(changes, dict) and set(changes) <= valid_keys, 'Unreviewed comparison controls')
        for key, values in changes.items():
            require(isinstance(values, list) and len(values) == 2 and values[0] != values[1],
                    'Expected explicit baseline/candidate control values: ' + key)
        allowed[field] = changes
    require(any(allowed.values()), 'No declared treatment difference')

    expected_limits = dict(rss_bytes=2 << 30, reserve_bytes=3 << 30, swap_growth_bytes=128 << 20,
                           admitted_payload_bytes=768 << 20, timeout_seconds=600., interval_seconds=.05,
                           max_processes=32)
    if 'guard_limits' in attestation:
        limits = attestation['guard_limits']
        require(isinstance(limits, dict) and set(limits) == set(expected_limits), 'Unexpected guard controls')
        require(all(limits[k] == expected_limits[k] for k in expected_limits if k != 'timeout_seconds'),
                'Guard safety limits changed')
        numeric(limits['timeout_seconds'], 'guard timeout', positive=True)
        expected_limits = limits

    jobs, runs = campaign.get('plan'), campaign.get('runs')
    require(isinstance(jobs, list) and isinstance(runs, list) and jobs and len(jobs) == len(runs),
            'Missing or incomplete planned workers')
    require([job.get('id') for job in jobs] == [run.get('id') for run in runs], 'Run order differs from plan')
    require(len({job['id'] for job in jobs}) == len(jobs), 'Duplicate worker IDs')
    pairs, rows, common_identity = {}, [], None

    def vector(step):
        artifact = step.get('logits_artifact')
        require(isinstance(artifact, dict), 'Missing raw logits artifact')
        relative = artifact.get('path')
        require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'Logits path must be relative')
        path = (folder / relative).resolve()
        require(path.is_relative_to(folder), 'Logits artifact escapes evidence folder')
        require(artifact.get('dtype') == 'float32', 'Only little-endian float32 logits are supported')
        shape = artifact.get('shape')
        require(isinstance(shape, list) and shape and all(type(n) is int and n > 0 for n in shape),
                'Invalid logits shape')
        raw = read_bytes(path)
        require(len(raw) == math.prod(shape) * 4, 'Logits artifact length differs from shape')
        actual_hash = hashlib.sha256(raw).hexdigest()
        require(actual_hash == digest(artifact.get('sha256'), 'logits artifact'), 'Raw logits digest differs')
        require(actual_hash == step.get('logits_sha256'), 'Logits digest differs from step')
        return raw, shape

    for job, run in zip(jobs, runs):
        label = job['id']
        require(Path(label).name == label and label not in ('.', '..'), 'Invalid worker filename')
        pair_id, variant = pair_identity(job)
        require(run.get('status') == 'completed' and type(run.get('exit_code')) is int and run['exit_code'] == 0,
                'Failed or incomplete worker: ' + label)
        worker = read_json(folder / (label + '.json'))
        guard = read_json(folder / (label + '.guard.json'))
        command = run.get('command')
        require(isinstance(command, list) and len(command) >= 2, 'Worker command missing')
        require(worker.get('command') == command[1:] + ['--worker'] and guard.get('command') == command + ['--worker'],
                'Worker/guard command chain differs')
        require(Path(argument(command, '--output')).name == label + '.json', 'Worker output command mismatch')
        require(Path(run.get('guard_file', '')).name == label + '.guard.json', 'Guard file mismatch')
        for flag, expected in (('--engine-root', job['engine_root']), ('--tasks', job['tasks']), ('--label', label)):
            require(argument(command, flag) == expected, 'Plan command differs: ' + flag)
        controls = job.get('args', [])
        require(isinstance(controls, list) and (not controls or command[-len(controls):] == controls),
                'Job controls differ from actual command')
        manifest = manifests.get(job['engine_root'])
        require(isinstance(manifest, dict), 'Unknown engine root')
        require(argument(command, '--expected-engine-head') == manifest['git_head']
                and json.loads(argument(command, '--expected-engine-sources-json')) == manifest['source_sha256']
                and json.loads(argument(command, '--expected-protocol-json')) == protocol,
                'Worker frozen inputs differ from campaign')
        require(worker.get('status') == 'completed' and worker.get('label') == label, 'Worker output incomplete')
        require(worker.get('engine_root') == job['engine_root'] and worker.get('source_sha256') == manifest['source_sha256']
                and worker.get('git_head') == worker.get('git_head_after') == manifest['git_head'], 'Worker source drift')
        for flag in ('source_unchanged', 'protocol_unchanged', 'model_stat_unchanged', 'engine_head_unchanged'):
            require(worker.get(flag) is True, 'Worker identity changed: ' + flag)
        tasks = campaign.get('tasks_by_job', {}).get(label)
        require(isinstance(tasks, list) and len(tasks) == 1 and worker.get('tasks_snapshot') == tasks,
                'Expected one frozen fresh-worker task')
        require(worker.get('protocol_sha256') == protocol and worker.get('harness_sha256') == protocol.get(command[1])
                and worker.get('tasks_sha256') == protocol.get(job['tasks']), 'Worker protocol digest mismatch')
        require(type(guard.get('exit_code')) is int and guard['exit_code'] == 0 and 'violation' in guard
                and guard['violation'] is None and not guard.get('exception') and not guard.get('cleanup_error'),
                'Independent guard failed')
        require(guard.get('limits') == expected_limits, 'Guard safety limits changed')
        require(natural(guard.get('sample_count'), 'guard samples') > 0, 'Guard did not sample')
        require(0 < natural(guard.get('sampled_peak_rss_bytes'), 'peak RSS') <= expected_limits['rss_bytes'],
                'RSS sampling missing or limit crossed')
        require(natural(guard.get('minimum_available_bytes'), 'system available') >= expected_limits['reserve_bytes'],
                'System reserve crossed')
        require(natural(guard.get('maximum_swap_growth_bytes'), 'swap growth') <= expected_limits['swap_growth_bytes'],
                'Swap growth limit crossed')
        require(guard.get('peak_pressure_level') == 1, 'Elevated or missing memory pressure')
        require(0 < natural(guard.get('sampled_peak_processes'), 'owned processes') <= expected_limits['max_processes'],
                'Process sampling missing or limit crossed')
        for side in ('before', 'after'):
            snapshot = guard.get(side, {})
            for field in ('physical_bytes', 'available_bytes', 'swap_used_bytes'):
                natural(snapshot.get(field), 'snapshot ' + field)
            require(snapshot.get('pressure_level') == 1, 'Guard snapshot pressure is not normal')
        require(guard['before']['available_bytes'] >= expected_limits['reserve_bytes'] + expected_limits['admitted_payload_bytes'],
                'Guard admission lacked reserve plus payload')
        if run.get('launch_wait_status') not in (None, 'disabled'):
            require(run['launch_wait_status'] == 'ready', 'Resource-window admission failed')
            policy = run.get('launch_wait_policy', {})
            require(policy.get('admission_limits') == expected_limits and policy.get('required_pressure_level') == 1,
                    'Resource-window policy changed guard safety limits')
            samples = run.get('launch_wait_samples', [])
            require(samples and samples[-1].get('ready') is True and samples[-1].get('observed_stable_seconds', -1) >= 5.,
                    'Missing stable resource window')
        identity = {field: worker.get(field) for field in ('model', 'model_config', 'hardware', 'versions', 'cache_regime')}
        require(all(value is not None for value in identity.values()), 'Machine/runtime/model identity missing')
        if common_identity is None:
            common_identity = identity
        require(identity == common_identity, 'Model, machine, runtime or cache regime differs')
        require(worker['model'].get('bytes') == model_bytes and model_bytes > guard['before']['physical_bytes'],
                'Model size differs or checkpoint is not larger than physical memory')
        require(argument(command, '--model') == worker['model'].get('path'), 'Model command path differs')
        require(worker.get('physical_ssd_bandwidth_certified') is False, 'Unsupported physical SSD certification')
        options = worker.get('options', {})
        require(options.get('reserve_bytes') == 3 << 30 and options.get('dtype') == 'float32'
                and options.get('device') == 'gpu' and options.get('prefetch') is False,
                'Precision/device/reserve/prefetch differs from reviewed controls')
        turns = worker.get('turns')
        require(isinstance(turns, list) and len(turns) == 1, 'Expected one worker turn')
        turn, task = turns[0], tasks[0]
        require(all(turn.get(key) == task.get(key) for key in ('id', 'prompt', 'max_new_tokens')), 'Turn differs from task')
        require(turn.get('status') == 'completed' and turn.get('stop_reason') == 'eos', 'Output did not complete at EOS')
        require(isinstance(turn.get('text'), str) and isinstance(turn.get('raw_text'), str), 'Complete output text missing')
        supplied, tokens, stops = (turn.get(field) for field in ('full_input_ids', 'generated_ids', 'eos_token_ids'))
        require(isinstance(supplied, list) and supplied and isinstance(tokens, list) and len(tokens) >= 2
                and isinstance(stops, list) and stops and len(tokens) <= task['max_new_tokens'], 'Invalid token arrays')
        for token in supplied + tokens + stops:
            natural(token, 'token ID')
        require(tokens[-1] in stops and not any(token in stops for token in tokens[:-1]), 'Invalid EOS termination')
        require(turn.get('full_output_ids') == supplied + tokens and turn.get('partial_ids') == tokens
                and turn.get('appended_input_ids') == supplied and turn.get('reused_prefix_tokens') == 0,
                'Fresh-worker emitted sequence differs')
        params = turn.get('generation_parameters', {})
        require(params.get('temperature') == 0 and params.get('do_sample') is False
                and params.get('max_new_tokens') == task['max_new_tokens'], 'Sampling controls differ')
        require(turn.get('finite_logits_checked') is True, 'Finite logits not checked')
        digest(turn.get('routing_sha256'), 'complete per-token routing')
        routing = turn.get('routing_by_layer_position')
        require(isinstance(routing, dict) and routing, 'Complete per-token routing records missing')
        encoded_routing = json.dumps(routing, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()
        require(hashlib.sha256(encoded_routing).hexdigest() == turn['routing_sha256'],
                'Complete routing digest differs from raw records')
        expected_positions = {str(index) for index in range(len(supplied) + len(tokens) - 1)}
        config = worker.get('model_config', {})
        layer_count = natural(config.get('num_hidden_layers'), 'model layer count')
        expert_count = natural(config.get('num_experts'), 'model expert count')
        top_k = natural(config.get('num_experts_per_tok'), 'active expert count')
        require(layer_count > 0 and 0 < top_k <= expert_count, 'Invalid model routing configuration')
        require(set(routing) == {str(index) for index in range(layer_count)}, 'Routing does not cover every model layer')
        for layer, positions in routing.items():
            require(isinstance(layer, str) and layer.isdecimal() and isinstance(positions, dict)
                    and set(positions) == expected_positions, 'Incomplete routing positions')
            for experts in positions.values():
                require(isinstance(experts, list) and len(experts) == top_k, 'Wrong active expert count')
                for expert in experts:
                    natural(expert, 'expert ID')
                    require(expert < expert_count, 'Routed expert ID is outside checkpoint range')
                require(len(set(experts)) == len(experts), 'Repeated expert in one route')
        steps, times = turn.get('step_checks'), turn.get('token_seconds')
        require(isinstance(steps, list) and len(steps) == len(tokens) and isinstance(times, list) and len(times) == len(tokens),
                'Missing per-token checks or times')
        for index, step in enumerate(steps):
            require(step.get('position') == len(supplied) + index, 'Causal/cache position differs')
            digest(step.get('logits_sha256'), 'step logits')
            digest(step.get('routes_sha256'), 'step routes')
            natural(step.get('logical_bytes'), 'step logical reads')
            numeric(step.get('forward_seconds'), 'forward time', positive=True)
            if step.get('process_storage_read_bytes') is not None:
                natural(step['process_storage_read_bytes'], 'OS-accounted reads')
        for value in times:
            numeric(value, 'per-token seconds', positive=True)
        require(turn.get('native_cache_position') == steps[-1]['position'], 'Final cache position differs')
        require(sum(step['logical_bytes'] for step in steps) == turn.get('logical_read_bytes'), 'Logical reads do not reconcile')
        require(turn['logical_read_bytes'] > 0, 'No model weight reads recorded')
        total = numeric(turn.get('total_seconds'), 'generation time', positive=True)
        ttft = numeric(turn.get('ttft_seconds'), 'first-token latency', positive=True)
        elapsed = numeric(guard.get('elapsed_seconds'), 'guard elapsed time', positive=True)
        require(total > ttft and elapsed >= numeric(worker.get('worker_wall_seconds'), 'worker time', positive=True),
                'Impossible timing order')
        decode_seconds = total - ttft
        decode_rate = (len(tokens) - 1) / decode_seconds
        require(math.isclose(decode_rate, numeric(turn.get('decode_tokens_per_second'), 'reported decode'),
                             rel_tol=1e-8, abs_tol=1e-8), 'Reported decode disagrees with independent calculation')
        os_reads = turn.get('process_storage_read_bytes')
        if os_reads is not None:
            natural(os_reads, 'worker OS-accounted reads')
        row = dict(label=label, pair_id=pair_id, variant=variant, input_tokens=len(supplied), generated_tokens=len(tokens),
                   ttft_seconds=ttft, generation_seconds=total, decode_seconds=decode_seconds,
                   decode_tokens=len(tokens) - 1, decode_tok_s=decode_rate, guard_seconds=elapsed,
                   logical_read_bytes=turn['logical_read_bytes'], first_forward_logical_bytes=steps[0]['logical_bytes'],
                   process_storage_read_bytes=os_reads, peak_rss_bytes=guard['sampled_peak_rss_bytes'],
                   minimum_available_bytes=guard['minimum_available_bytes'], swap_initial_bytes=guard['before']['swap_used_bytes'],
                   swap_growth_bytes=guard['maximum_swap_growth_bytes'])
        rows.append(row)
        pair = pairs.setdefault(pair_id, {})
        require(variant not in pair, 'Duplicate variant in pair')
        pair[variant] = (worker, turn, row)

    require(len(pairs) >= minimum_pairs and all(set(pair) == {'baseline', 'candidate'} for pair in pairs.values()),
            'Missing complete baseline/candidate pairs')
    pair_order = list(pairs)
    orders = [[row['variant'] for row in rows if row['pair_id'] == pair_id] for pair_id in pair_order]
    require(all(order == (['baseline', 'candidate'] if index % 2 == 0 else ['candidate', 'baseline'])
                for index, order in enumerate(orders)), 'Pair order is not alternating AB/BA')
    paired = []
    for pair_id, pair in pairs.items():
        baseline, bt, br = pair['baseline']
        candidate, ct, cr = pair['candidate']
        require(baseline.get('tasks_snapshot') == candidate.get('tasks_snapshot'), 'Paired frozen tasks differ')
        for field, differences in allowed.items():
            before, after = dict(baseline.get(field, {})), dict(candidate.get(field, {}))
            for key, values in differences.items():
                require([before.pop(key, None), after.pop(key, None)] == values, 'Treatment differs: ' + field + '/' + key)
            require(before == after, 'Undeclared comparison difference: ' + field)
        for field in ('full_input_ids', 'appended_input_ids', 'generated_ids', 'full_output_ids', 'eos_token_ids',
                      'generation_parameters', 'native_cache_position', 'routing_sha256', 'text', 'raw_text'):
            require(bt.get(field) == ct.get(field), 'Paired output/routing differs: ' + field)
        max_abs, max_rel, elements = 0., 0., 0
        logits_exact = True
        for index, (bs, cs) in enumerate(zip(bt['step_checks'], ct['step_checks'])):
            # First-forward routes can be grouped differently across prefill chunks;
            # complete routing_sha256 above is normalized by layer and position.
            require(bs['position'] == cs['position'], 'Paired causal position differs')
            if index:
                require(bs['routes_sha256'] == cs['routes_sha256'], 'Paired decode routes differ')
            braw, bshape = vector(bs)
            craw, cshape = vector(cs)
            require(bshape == cshape, 'Paired logits shapes differ')
            logits_exact &= braw == craw
            if mode == 'exact':
                require(braw == craw, 'Logits are not bitwise exact')
            for (bv,), (cv,) in zip(struct.iter_unpack('<f', braw), struct.iter_unpack('<f', craw)):
                require(math.isfinite(bv) and math.isfinite(cv), 'Raw logits contain NaN or infinity')
                delta = abs(bv - cv)
                require(delta <= atol + rtol * abs(bv), 'Raw logits exceed predeclared numerical tolerance')
                max_abs = max(max_abs, delta)
                max_rel = max(max_rel, delta / max(abs(bv), 1e-30))
                elements += 1
        paired.append(dict(pair_id=pair_id, input_tokens=br['input_tokens'],
                           tokens_routes_positions_text_exact=True, logits_bitwise_exact=logits_exact,
                           logits_elements_checked=elements, logits_max_absolute_error=max_abs,
                           logits_max_relative_error=max_rel, ttft_speedup=br['ttft_seconds'] / cr['ttft_seconds'],
                           ttft_reduction_seconds=br['ttft_seconds'] - cr['ttft_seconds'],
                           decode_speedup=cr['decode_tok_s'] / br['decode_tok_s'],
                           logical_read_ratio=cr['logical_read_bytes'] / br['logical_read_bytes']))
    aggregate = {}
    for variant in ('baseline', 'candidate'):
        selected = [row for row in rows if row['variant'] == variant]
        decode_seconds = sum(row['decode_seconds'] for row in selected)
        aggregate[variant] = dict(workers=len(selected), ttft_seconds=describe([row['ttft_seconds'] for row in selected]),
            decode_tok_s=sum(row['decode_tokens'] for row in selected) / decode_seconds,
            decode_seconds=decode_seconds, logical_read_bytes=sum(row['logical_read_bytes'] for row in selected),
            first_forward_logical_bytes=sum(row['first_forward_logical_bytes'] for row in selected),
            process_storage_read_bytes=None if any(row['process_storage_read_bytes'] is None for row in selected)
                else sum(row['process_storage_read_bytes'] for row in selected),
            peak_rss_bytes=max(row['peak_rss_bytes'] for row in selected),
            minimum_available_bytes=min(row['minimum_available_bytes'] for row in selected),
            maximum_swap_growth_bytes=max(row['swap_growth_bytes'] for row in selected))
    for path, expected in evidence_hashes.items():
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Evidence changed during verification: ' + str(path))
    return dict(status='verified', fixture=campaign.get('fixture', False), worker_count=len(rows), pair_count=len(pairs),
        numerical_policy=numerical, all_tokens_routes_positions_text_exact=True,
        all_logits_bitwise_exact=all(pair['logits_bitwise_exact'] for pair in paired),
        paired_ttft_speedup=describe([pair['ttft_speedup'] for pair in paired]),
        aggregate=aggregate, pairs=paired, workers=rows, checkpoint_sha256=before_hash,
        checkpoint_integrity_scope='Checks recorded full-file before/after SHA256; does not reopen checkpoint',
        storage_scope='Logical read requests and OS-accounted process reads; no cold SSD or NAND claim',
        inputs=[dict(path=str(path), sha256=value) for path, value in evidence_hashes.items()])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--integrity', type=Path)
    parser.add_argument('--check-current-inputs', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify(args.campaign, args.integrity, check_current=args.check_current_inputs)
        code = 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(status='rejected', error=f'{type(error).__name__}: {error}')
        code = 2
    result['checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('inputs', 'workers', 'pairs', 'aggregate')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
