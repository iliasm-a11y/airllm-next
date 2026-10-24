"""Independent stdlib-only verification of the completed primary raw evidence."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--campaign', type=Path, required=True)
parser.add_argument('--summary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
RAW = args.campaign.resolve()
SUMMARY = args.summary.resolve()
OUTPUT = args.output.resolve()
inputs, failures = {}, []


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    raw = path.read_bytes()
    inputs[str(path.resolve())] = {'sha256': digest(raw), 'bytes': len(raw)}
    return json.loads(raw)


def check(condition, message):
    if not condition:
        failures.append(message)


def stat(values):
    values = sorted(values)
    def quantile(q):
        position = (len(values) - 1) * q
        first = math.floor(position)
        return values[first] + (values[math.ceil(position)] - values[first]) * (position - first)
    return dict(n=len(values), median=statistics.median(values), q1=quantile(.25), q3=quantile(.75),
                iqr=quantile(.75)-quantile(.25), min=min(values), max=max(values))


def compare(actual, expected, label):
    if isinstance(actual, dict):
        for key, value in actual.items():
            if key not in expected:
                failures.append(label + '.' + key + ': missing from summary')
            else:
                compare(value, expected[key], label + '.' + key)
    elif isinstance(actual, float):
        check(isinstance(expected, (float, int)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9),
              f'{label}: independent={actual!r}, summary={expected!r}')
    else:
        check(actual == expected, f'{label}: independent={actual!r}, summary={expected!r}')


campaign = read(RAW / 'campaign.json')
check(campaign['status'] == 'completed' and campaign['protocol_unchanged'] is True
      and campaign['engines_unchanged'] is True, 'Campaign did not finish with stable inputs')
check(campaign['engine_manifests'] == campaign['engine_manifests_after'], 'Campaign engine manifests changed')
plan = {job['id']: job for job in campaign['plan']}
runs = {run['id']: run for run in campaign['runs']}
check(len(plan) == len(runs) == 12, 'Expected exactly twelve planned/completed workers')
rows, pairs, references = [], {}, {}
for label in sorted(plan):
    match = re.fullmatch(r'(short|explanation|long_context)\.r([01])\.(baseline|candidate)', label)
    check(match is not None, 'Unexpected worker ID: ' + label)
    task, repeat, variant = match[1], int(match[2]), match[3]
    raw = read(RAW / (label + '.json'))
    guard = read(RAW / (label + '.guard.json'))
    job, run = plan[label], runs[label]
    expected_engine = campaign['engine_manifests'][job['engine_root']]
    check(run['status'] == 'completed' and run['exit_code'] == 0, label + ': unsuccessful campaign run')
    check(raw['status'] == 'completed' and all(raw[key] is True for key in
          ('source_unchanged','protocol_unchanged','model_stat_unchanged','engine_head_unchanged')),
          label + ': worker incomplete or provenance drift')
    check(raw['engine_root'] == job['engine_root'] and raw['git_head'] == raw['git_head_after']
          == expected_engine['git_head'] and raw['source_sha256'] == expected_engine['source_sha256'],
          label + ': engine provenance mismatch')
    check(raw['protocol_sha256'] == campaign['protocol_sha256'], label + ': protocol mismatch')
    check(raw['tasks_sha256'] == campaign['protocol_sha256'][job['tasks']], label + ': task hash mismatch')
    check(raw['harness_sha256'] == next(value for path, value in campaign['protocol_sha256'].items()
          if path.endswith('/benchmarks/benchmark_research51.py')), label + ': harness hash mismatch')
    check(raw['tasks_snapshot'] == campaign['tasks_by_job'][label], label + ': requested task changed')
    check(guard['exit_code'] == 0 and type(guard['exit_code']) is int and guard['violation'] is None
          and not guard.get('exception') and not guard.get('cleanup_error') and guard['sample_count'] > 0,
          label + ': invalid independent guard')
    check(guard['command'] == [*run['command'], '--worker'] and raw['command'] == guard['command'][1:],
          label + ': command mismatch')
    check(guard['sampled_peak_rss_bytes'] <= guard['limits']['rss_bytes']
          and guard['sampled_peak_processes'] <= guard['limits']['max_processes']
          and guard['minimum_available_bytes'] >= guard['limits']['reserve_bytes']
          and guard['peak_pressure_level'] == 1, label + ': resource threshold exceeded')
    swap_states = [raw['before']['swap_used_bytes'], raw['after']['swap_used_bytes'],
                   guard['before']['swap_used_bytes'], guard['after']['swap_used_bytes'], guard['maximum_swap_growth_bytes']]
    check(swap_states == [0] * 5, label + ': nonzero sampled or endpoint system swap')
    check(len(raw['turns']) == len(raw['tasks_snapshot']) == 1, label + ': unexpected task count')
    turn = raw['turns'][0]
    check(all(turn[key] == raw['tasks_snapshot'][0][key] for key in ('id','prompt','max_new_tokens')),
          label + ': actual prompt differs')
    n = len(turn['generated_ids']); steps = turn['step_checks']; prompt_n = len(turn['full_input_ids'])
    check(turn['status'] == 'completed' and turn['stop_reason'] == 'eos'
          and turn['generated_ids'][-1] in turn['eos_token_ids'] and turn['finite_logits_checked'] is True,
          label + ': unfinished/nonfinite output')
    check(len(steps) == len(turn['token_seconds']) == n
          and [step['position'] for step in steps] == list(range(prompt_n, prompt_n + n))
          and turn['native_cache_position'] == steps[-1]['position'], label + ': step/position mismatch')
    check(turn['full_output_ids'] == turn['full_input_ids'] + turn['generated_ids']
          and turn['partial_ids'] == turn['generated_ids'], label + ': output token journal mismatch')
    check(all(re.fullmatch('[0-9a-f]{64}', step[field]) for step in steps
              for field in ('logits_sha256','routes_sha256')), label + ': malformed exactness hash')
    for key in ('model','hardware','versions','cache_regime'):
        reference = references.setdefault(key, raw[key])
        check(raw[key] == reference, label + ': environment/model identity changed: ' + key)
    references.setdefault('guard_limits', guard['limits'])
    check(guard['limits'] == references['guard_limits'], label + ': guard limits changed')
    options = {key:value for key,value in raw['options'].items() if key != 'zero_copy_weights'}
    references.setdefault('common_options', options)
    check(options == references['common_options'], label + ': common runtime options changed')
    decode_seconds = turn['total_seconds'] - turn['ttft_seconds']
    check(math.isclose((n - 1) / decode_seconds, turn['decode_tokens_per_second'], rel_tol=1e-12),
          label + ': reported streaming rate disagrees with generation-minus-TTFT interval')
    row = dict(task=task, repeat=repeat, variant=variant, label=label, generated=n, decode=n-1,
               stream_seconds=decode_seconds, forward_seconds=sum(turn['token_seconds'][1:]),
               generation_seconds=turn['total_seconds'], elapsed_seconds=guard['elapsed_seconds'],
               worker_seconds=raw['worker_wall_seconds'], ttft=turn['ttft_seconds'],
               rss=guard['sampled_peak_rss_bytes'], mlx=turn['metrics']['allocator']['peak_bytes'],
               available=guard['minimum_available_bytes'], reserve=guard['limits']['reserve_bytes'],
               logical=turn['logical_read_bytes'], physical=turn['process_storage_read_bytes'],
               decode_logical=sum(step['logical_bytes'] for step in steps[1:]),
               decode_physical=sum(step['process_storage_read_bytes'] for step in steps[1:]),
               hits=turn['metrics']['cache'].get('hits', 0), misses=turn['metrics']['cache'].get('misses', 0))
    rows.append(row)
    pairs.setdefault((task, repeat), {})[variant] = dict(raw=raw, turn=turn, row=row)


def aggregate(items):
    generated, decoded = sum(r['generated'] for r in items), sum(r['decode'] for r in items)
    stream, forward = sum(r['stream_seconds'] for r in items), sum(r['forward_seconds'] for r in items)
    elapsed = sum(r['elapsed_seconds'] for r in items)
    return dict(workers=len(items), generated_tokens_including_eos=generated, decode_tokens_including_final_eos=decoded,
                decode_step_seconds=forward, stream_decode_seconds=stream, decode_tok_s=decoded/stream,
                forward_only_decode_tps=decoded/forward, end_to_end_tok_s=generated/elapsed, guard_elapsed_seconds=elapsed,
                generation_tok_s=generated/sum(r['generation_seconds'] for r in items),
                worker_wall_tok_s_excluding_imports=generated/sum(r['worker_seconds'] for r in items),
                ttft_seconds=stat([r['ttft'] for r in items]), peak_owned_rss_bytes_sampled=max(r['rss'] for r in items),
                peak_mlx_allocator_bytes=max(r['mlx'] for r in items),
                minimum_available_bytes_sampled=min(r['available'] for r in items),
                minimum_reserve_headroom_bytes_sampled=min(r['available']-r['reserve'] for r in items),
                maximum_system_swap_growth_bytes_sampled=0, logical_read_bytes=sum(r['logical'] for r in items),
                process_storage_read_bytes=sum(r['physical'] for r in items),
                decode_logical_read_bytes_per_token=sum(r['decode_logical'] for r in items)/decoded,
                decode_process_storage_read_bytes_per_token=sum(r['decode_physical'] for r in items)/decoded,
                cache_hits=sum(r['hits'] for r in items), cache_misses=sum(r['misses'] for r in items))


pair_rows = []
for (task, repeat), group in sorted(pairs.items()):
    check(set(group) == {'baseline','candidate'}, f'{task}.r{repeat}: missing variant')
    a, b = group['baseline'], group['candidate']
    equality = {field: a['turn'][field] == b['turn'][field] for field in
                ('prompt','full_input_ids','appended_input_ids','generated_ids','full_output_ids','text','raw_text','eos_token_ids')}
    equality['logits_routes_positions'] = [(s['logits_sha256'],s['routes_sha256'],s['position']) for s in a['turn']['step_checks']] == [
        (s['logits_sha256'],s['routes_sha256'],s['position']) for s in b['turn']['step_checks']]
    check(all(equality.values()), f'{task}.r{repeat}: exactness mismatch')
    ar, br = a['row'], b['row']
    pair_rows.append(dict(task=task, repeat=repeat, equality=equality,
                         decode_speedup=(br['decode']/br['stream_seconds'])/(ar['decode']/ar['stream_seconds']),
                         forward_only_decode_speedup=(br['decode']/br['forward_seconds'])/(ar['decode']/ar['forward_seconds']),
                         end_to_end_speedup=(br['generated']/br['elapsed_seconds'])/(ar['generated']/ar['elapsed_seconds']),
                         ttft_speedup=ar['ttft']/br['ttft']))

aggregates = {variant: aggregate([r for r in rows if r['variant']==variant]) for variant in ('baseline','candidate')}
decode_gain = aggregates['candidate']['decode_tok_s']/aggregates['baseline']['decode_tok_s']
e2e_gain = aggregates['candidate']['end_to_end_tok_s']/aggregates['baseline']['end_to_end_tok_s']
summary = read(SUMMARY)
compare(dict(pair_count=len(pairs), worker_count=len(rows), prompt_count=len({r['task'] for r in rows}),
             repeats_per_prompt=2, all_tokens_logits_routes_positions_exact=all(all(p['equality'].values()) for p in pair_rows),
             all_complete_responses_exact=all(p['equality']['text'] and p['equality']['raw_text'] for p in pair_rows),
             aggregate=aggregates, aggregate_decode_speedup=decode_gain, aggregate_end_to_end_speedup=e2e_gain,
             paired_decode_speedup=stat([p['decode_speedup'] for p in pair_rows]),
             paired_end_to_end_speedup=stat([p['end_to_end_speedup'] for p in pair_rows]),
             paired_forward_only_decode_speedup=stat([p['forward_only_decode_speedup'] for p in pair_rows]),
             paired_ttft_speedup=stat([p['ttft_speedup'] for p in pair_rows]),
             paired_decode_wins=sum(p['decode_speedup']>1 for p in pair_rows)), summary, 'summary')
for task in sorted({r['task'] for r in rows}):
    for variant in ('baseline','candidate'):
        compare(aggregate([r for r in rows if r['task']==task and r['variant']==variant]),
                summary['by_prompt'][task][variant], f'summary.by_prompt.{task}.{variant}')
for pair in pair_rows:
    existing = next(item for item in summary['pairs'] if item['task']==pair['task'] and item['repeat']==pair['repeat'])
    compare({key:pair[key] for key in ('task','repeat','decode_speedup','forward_only_decode_speedup','end_to_end_speedup','ttft_speedup')},
            existing, 'summary.pair')
summary_inputs = {item['path']:item for item in summary['inputs']}
for path, info in inputs.items():
    if path != str(SUMMARY.resolve()):
        check(path in summary_inputs and summary_inputs[path]['sha256'] == info['sha256'], 'Analyzer input hash mismatch: ' + path)

quality = []
for task in sorted({r['task'] for r in rows}):
    first = pairs[(task,0)]['candidate']['turn']; second = pairs[(task,1)]['candidate']['turn']
    check(first['text'] == second['text'] and first['generated_ids'] == second['generated_ids'], task + ': repeat output differs')
    quality.append(dict(task=task, prompt=first['prompt'], text=first['text'], whitespace_words=len(first['text'].split()),
                        input_tokens=len(first['full_input_ids']), generated_tokens_including_eos=len(first['generated_ids']),
                        repeats_text_and_tokens_exact=True))
result = dict(kind='independent_stdlib_raw_primary_verification', status='passed' if not failures else 'failed',
              calculation_implementation_sha256=digest(Path(__file__).read_bytes()), implementation_path=str(Path(__file__).resolve()),
              scope=dict(workers=12,pairs=6,prompts=3,repetitions_per_prompt=2,
                         independent_of_analyzer=True, analyzer_code_imported=False, model_or_engine_executed=False,
                         current_model_and_current_engine_sources_read=False,
                         evidence_hashes_verified=True, system_swap_scope='endpoint and sampled system-wide counters'),
              raw_input_hashes=inputs, recorded_engine_heads={root:data['git_head'] for root,data in campaign['engine_manifests'].items()},
              aggregate=aggregates, aggregate_decode_speedup=decode_gain, aggregate_decode_gain_percent=100*(decode_gain-1),
              aggregate_end_to_end_speedup=e2e_gain, aggregate_end_to_end_gain_percent=100*(e2e_gain-1),
              pairs=pair_rows, all_guard_violations_none=True, all_endpoint_and_sampled_swap_zero=True,
              summary_comparison='matches within rel_tol=1e-12 / abs_tol=1e-9' if not failures else 'mismatch',
              failures=failures, quality_observations=quality,
              quality_limits=[
                  'Explanation is 106 whitespace-delimited words despite a maximum of 100.',
                  'Swap is a generic possibility for some oversized workloads; the answer does not describe the measured selective-streaming engine with observed zero swap growth.',
                  'The task called long_context uses only 233 template-inclusive input tokens, without a reused prefix; this is not evidence of genuinely long-context or multiturn quality.',
                  'Identical complete responses certify preservation on these prompts, not general semantic quality.'
              ],
              measurement_limits=[
                  'EOS tokens are included in generated and final decode token totals.',
                  'Streaming seconds independently reconstructed as total_seconds - ttft_seconds; includes callback/journal instrumentation.',
                  'End-to-end rate uses parent guard elapsed including fresh worker startup/imports/supervision.',
                  'Fresh workers do not control OS page-cache state; storage counters are not certified physical SSD bandwidth.',
                  'Model integrity uses recorded stat identity, not a fresh complete model checksum.',
                  'Exactness covers recorded per-token logits/routes hashes, token IDs and positions; no full KV payload comparison was recorded.',
                  'Six paired observations across three prompts are descriptive and do not establish population or thermal confidence.',
                  'Expert profiler and stability runs are outside this independent primary verification.'
              ])
with OUTPUT.open('x',encoding='utf-8') as handle:
    json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
print(json.dumps({'status':result['status'],'workers':len(rows),'pairs':len(pairs),'decode_gain_percent':result['aggregate_decode_gain_percent'],
                  'end_to_end_gain_percent':result['aggregate_end_to_end_gain_percent'],'failures':failures,'output':str(OUTPUT)},ensure_ascii=False))
if failures:
    sys.exit(1)
