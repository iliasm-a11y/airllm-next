"""Evidence-analysis regressions: guards, actual units, paired controls and nulls."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / 'benchmarks' / 'synthesize_multitoken.py'
_SPEC = importlib.util.spec_from_file_location('multitoken_synthesis', _PATH)
synthesis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(synthesis)


def options():
    return dict(cache_bytes=40, io_bytes=30, kv_bytes=20, workspace_bytes=10)


def row(repeat, variant='autoregressive', *, adaptive=False):
    units = (2, 8)[repeat]
    reference = variant == 'autoregressive'
    seconds = units if reference else units / (4 if adaptive else 2)
    runtime = dict(tokens_generated=units, prefill_tokens=2,
                   encoded_file_bytes_per_token_including_prefill=100)
    result = dict(kind='baseline' if reference else 'speculation', case=0, repeat=repeat,
                  variant=variant, new_tokens=units, seconds=seconds,
                  tokens=list(range(units + 2)), runtime=runtime,
                  tokens_match_baseline=True, io_delta={'logical_bytes': units * 100},
                  expert_delta={'batches': 2, 'distinct_expert_sum': 8},
                  physical_read_bytes=0, kv_peak_bytes=8)
    if reference:
        result['options'] = options()
    else:
        result.update(depth=32 if adaptive else 2, adaptive=adaptive, prefetch=False,
                      spec_prefetch=False, target_options=options())
        runtime['speculative'] = dict(tokens_generated=units, proposals=(2, 8)[repeat],
            accepted_draft_tokens=(1, 6)[repeat], encoded_file_bytes=units * 100,
            total_logical_bytes=units * 110, total_encoded_file_bytes=units * 110,
            rounds=1, target_forward_calls=3, target_forward_positions=units + 2,
            draft_seconds=.1, verification_seconds=.2, rollback_tokens=0)
    return result


def report():
    return dict(schema_version=1, phase='generation', primary_campaign=True,
        eos_policy='model_config_stop', source_sha256={'engine.py': '0' * 64},
        source_unchanged=True, failures=[], baseline_options=options(),
        arguments=dict(prompts=1, repeats=2, depths=[2], drafts=['prompt']),
        records=[row(repeat, variant, adaptive=adaptive) for repeat in (0, 1)
                 for variant, adaptive in [('autoregressive', False), ('prompt', False), ('prompt', True)]]
                 + [dict(kind='sequence_seed', case=0, seconds=.000001, new_tokens=100000)])


def test_actual_unit_weighting_and_paired_medians(tmp_path):
    summary = synthesis.summarize_report(report(), tmp_path / 'generation-test.json')
    assert summary['accepted']
    fixed = next(group for group in summary['groups'] if group['variant_key'][1] == 2)
    assert fixed['pairs'] == 2
    assert fixed['median_paired_throughput_speedup'] == 2
    assert fixed['variant']['actual_units'] == 10
    assert fixed['variant']['total_logical_bytes_per_actual_unit'] == 110
    assert fixed['variant']['acceptance_rate_weighted_by_proposals'] == .7
    assert fixed['median_paired_os_accounted_storage_read_bytes_per_unit_reduction'] is None


@pytest.mark.parametrize('bad', ['source', 'missing_reference', 'duplicate', 'tokens', 'budget', 'unfinished'])
def test_invalid_campaign_never_emits_gains(tmp_path, bad):
    data = report()
    if bad == 'source':
        data['source_unchanged'] = False
    elif bad == 'missing_reference':
        data['records'].pop(0)
    elif bad == 'duplicate':
        data['records'].append(copy.deepcopy(data['records'][0]))
    elif bad == 'tokens':
        data['records'][1]['tokens'][-1] = 999
    elif bad == 'budget':
        data['records'][1]['target_options']['cache_bytes'] += 1
    elif bad == 'unfinished':
        data['completed'] = False
    summary = synthesis.summarize_report(data, tmp_path / 'generation-test.json')
    assert not summary['accepted']
    assert summary['errors']
    assert not summary['groups']


def test_unknown_draft_transport_remains_unknown(tmp_path):
    data = report()
    for item in data['records']:
        if item.get('kind') == 'speculation':
            item['runtime']['speculative']['total_logical_bytes'] = None
    summary = synthesis.summarize_report(data, tmp_path / 'generation-test.json')
    assert summary['accepted']
    for group in summary['groups']:
        assert group['variant']['total_logical_bytes_per_actual_unit'] is None
        assert group['median_paired_total_logical_bytes_per_unit_reduction'] is None


def test_primary_filter_does_not_count_exploratory_report_as_failure(tmp_path):
    primary, exploratory = report(), report()
    exploratory['primary_campaign'] = False
    files = [tmp_path / 'generation-primary.json', tmp_path / 'generation-exploratory.json']
    for path, data in zip(files, (primary, exploratory)):
        path.write_text(json.dumps(data))
    output = tmp_path / 'summary.json'
    assert synthesis.main([*(str(path) for path in files), '--primary-only', '--output', str(output)]) == 0
    result = json.loads(output.read_text())
    assert result['accepted_reports'] == 1
    assert result['skipped_reports'] == 1
    assert result['excluded_reports'] == 0


def test_batch_arrays_and_exactness_coverage_are_required(tmp_path):
    path = tmp_path / 'batch-test.json'
    path.with_suffix('.npz').write_bytes(b'raw fixture bytes')
    data = report()
    data.update(phase='batch', arguments=dict(prompts=1, repeats=1, batch_sizes=[2]),
        raw_arrays_sha256=synthesis.digest(path.with_suffix('.npz')), records=[], comparisons=[])
    for variant in ('tokenwise', 'serial_positions', 'union'):
        data['records'].append(dict(kind='batch', case=0, repeat=0, variant=variant,
            batch_size=2, positions=2, seconds=2 if variant == 'tokenwise' else 1,
            io_delta={'logical_bytes': 200}, expert_delta={}, kv_peak_bytes=8, physical_read_bytes=0))
        data['comparisons'].append(dict(case=0, repeat=0, variant=variant, batch_size=2,
            logits_close=True, argmax_equal=True, route_sets_equal=True))
    summary = synthesis.summarize_report(data, path)
    assert summary['accepted']
    assert len(summary['groups']) == 3
    assert all(group['actual_unit'] == 'verified_causal_position' for group in summary['groups'])
    path.with_suffix('.npz').write_bytes(b'altered')
    assert not synthesis.summarize_report(data, path)['accepted']


def test_cache_policies_use_matched_adaptive_controls(tmp_path):
    data = report()
    data.update(phase='cache', arguments=dict(prompts=1, repeats=2, policies=['adaptive', 'shared']), records=[])
    for repeat in (0, 1):
        for policy in ('adaptive', 'shared'):
            item = row(repeat)
            item.update(kind='cache', variant=policy)
            if policy == 'shared':
                item['seconds'] /= 1.5
            data['records'].append(item)
    summary = synthesis.summarize_report(data, tmp_path / 'cache-test.json')
    assert summary['accepted']
    group, = summary['groups']
    assert group['baseline_key'] == ['adaptive']
    assert group['median_paired_throughput_speedup'] == 1.5


def test_combination_prefetch_variants_remain_distinct(tmp_path):
    data = report()
    data.update(phase='combination', arguments=dict(prompts=1, repeats=2, drafts=['prompt']), records=[])
    for repeat in (0, 1):
        data['records'].append(row(repeat))
        for depth, adaptive, prefetch, spec in [(4, False, False, False), (4, False, True, False),
                                               (4, False, True, True), (32, True, True, False)]:
            item = row(repeat, 'prompt', adaptive=adaptive)
            item.update(depth=depth, prefetch=prefetch, spec_prefetch=spec)
            data['records'].append(item)
    summary = synthesis.summarize_report(data, tmp_path / 'combination-test.json')
    assert summary['accepted']
    assert len(summary['groups']) == 4
