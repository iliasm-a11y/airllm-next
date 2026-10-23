"""Mathematical/selection checks only; no corpus inference or performance test."""
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('benchmark_quantized_corpus_quality',
    Path(__file__).with_name('benchmark_quantized_corpus_quality.py'))
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_human_label_nll_ppl_and_kl_distinguish_quality_from_reference_agreement():
    reference = np.array([[2., 0.], [0., 2.]])
    actual = np.array([[0., 2.], [0., 2.]])
    metric = BENCHMARK.quality_metrics(reference, actual, [0, 1], row_chunk=1)
    assert metric['reference_mean_nll'] == pytest.approx(math.log1p(math.exp(-2)))
    assert metric['mean_nll_delta'] == pytest.approx(1.)
    assert metric['perplexity_ratio'] == pytest.approx(math.e)
    assert metric['mean_kl_reference_to_variant'] == pytest.approx(math.tanh(1.))
    assert metric['top1_agreement_with_original'] == .5
    assert metric['reference_human_next_token_accuracy'] == 1.
    assert metric['variant_human_next_token_accuracy'] == .5


def test_aggregation_weights_log_likelihoods_by_token_count_before_exponentiation():
    reference = np.array([[2., 0.], [0., 2.], [2., 0.]])
    actual = np.array([[0., 2.], [0., 2.], [2., 0.]])
    labels = np.array([0, 1, 0])
    first = BENCHMARK.quality_metrics(reference[:1], actual[:1], labels[:1])
    rest = BENCHMARK.quality_metrics(reference[1:], actual[1:], labels[1:])
    combined = BENCHMARK.aggregate([first, rest])
    direct = BENCHMARK.quality_metrics(reference, actual, labels)
    for key in ('reference_perplexity', 'variant_perplexity', 'mean_nll_delta',
                'mean_kl_reference_to_variant', 'top1_agreement_with_original'):
        assert combined[key] == pytest.approx(direct[key])
    assert combined['scored_tokens'] == 3


def test_log_softmax_remains_stable_and_rejects_invalid_labels_or_nonfinite_logits():
    reference = np.array([[10000., 10002.]])
    metric = BENCHMARK.quality_metrics(reference, reference, [1])
    assert metric['mean_kl_reference_to_variant'] == 0.
    assert metric['perplexity_ratio'] == 1.
    assert metric['reference_mean_nll'] == pytest.approx(math.log1p(math.exp(-2)))
    with pytest.raises(ValueError, match='labels'):
        BENCHMARK.quality_metrics(reference, reference, [2])
    with pytest.raises(ValueError, match='Nonfinite'):
        BENCHMARK.quality_metrics(reference, np.array([[np.nan, 1.]]), [1])


def test_predeclared_quantile_windows_are_disjoint_and_score_exactly_512_labels():
    rows = BENCHMARK.select_segments(list(range(1000)))
    assert [row['source_token_start'] for row in rows] == [45, 295, 545, 795]
    assert sum(row['scored_tokens'] for row in rows) == 512
    row = rows[0]
    inputs = row['token_ids'][:-1]
    labels = row['token_ids'][row['context_tokens']:]
    assert len(inputs) == 159 and len(labels) == 128
    # The last 128 prediction positions each target the immediately next token.
    assert [value+1 for value in inputs[-128:]] == labels


def test_prepared_manifest_detects_token_and_position_edits():
    manifest = dict(corpus=dict(source_sha256=BENCHMARK.SOURCE_SHA256,
        source_revision=BENCHMARK.REVISION), corpus_total_token_count=1000,
        segments=BENCHMARK.select_segments(list(range(1000))))
    BENCHMARK.validate_prepared_manifest(manifest, 4, 128, 32)
    edited = copy.deepcopy(manifest)
    edited['segments'][0]['token_ids'][0] += 1
    with pytest.raises(ValueError, match='digest'):
        BENCHMARK.validate_prepared_manifest(edited, 4, 128, 32)
    edited = copy.deepcopy(manifest)
    edited['segments'][1]['source_token_start'] += 1
    with pytest.raises(ValueError, match='positions'):
        BENCHMARK.validate_prepared_manifest(edited, 4, 128, 32)


def test_all_quality_variants_keep_f_budgets_and_fp32_accumulation(tmp_path):
    args = SimpleNamespace(device='cpu', cache_mib=256, pack_root=tmp_path)
    configs = [BENCHMARK.configuration(args, tmp_path/'metadata', variant) for variant in BENCHMARK.VARIANTS]
    for cfg in configs:
        assert cfg.dtype == 'float32'
        assert cfg.cache_bytes == 256*BENCHMARK.MiB
        assert cfg.io_bytes == cfg.kv_bytes == cfg.max_tensor_bytes == 32*BENCHMARK.MiB
        assert cfg.workspace_bytes == 256*BENCHMARK.MiB
        assert cfg.reserve_bytes == 1536*BENCHMARK.MiB
        assert cfg.prefetch is cfg.spec_prefetch is cfg.use_saved_tuning is False
        assert cfg.max_context == 512 and cfg.prefill_chunk == 64 and cfg.row_chunk == 2048
    assert configs[0].quantized_dir is None
    assert Path(configs[1].quantized_dir).name == 'quantized-granite-expert4'
    assert Path(configs[2].quantized_dir).name == 'quantized-granite-mixed'


@pytest.mark.parametrize('changed,expected', [
    ('cache_peak', 'cache'), ('io_peak', 'IO'), ('kv_peak', 'KV'),
    ('kv_bytes', 'KV'), ('kv_position', 'position'), ('workspace', 'workspace')])
def test_corpus_guard_checks_peaks_context_and_workspace(changed, expected):
    config = SimpleNamespace(cache_bytes=16, io_bytes=8, kv_bytes=12, workspace_bytes=24)
    cache = SimpleNamespace(bytes=8, peak_bytes=16)
    io = SimpleNamespace(peak_bytes=8)
    kv = SimpleNamespace(bytes=8, peak_bytes=12, position=159)
    workspace = [24]
    model = SimpleNamespace(reader=SimpleNamespace(cache=cache, io=io),
        expert_batch_metrics=lambda: {'totals': {'workspace_peak_bytes': workspace[0]}})
    output = SimpleNamespace(past_key_values=kv)
    assert BENCHMARK.budget_snapshot(model, output, config, 159)['kv_position'] == 159
    if changed == 'cache_peak': cache.peak_bytes = 17
    if changed == 'io_peak': io.peak_bytes = 9
    if changed == 'kv_peak': kv.peak_bytes = 13
    if changed == 'kv_bytes': kv.bytes = 13
    if changed == 'kv_position': kv.position = 160
    if changed == 'workspace': workspace[0] = 25
    with pytest.raises(AssertionError, match=expected):
        BENCHMARK.budget_snapshot(model, output, config, 159)
