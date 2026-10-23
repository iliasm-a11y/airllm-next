"""Small deterministic checks for lossy benchmark accounting, not model quality."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('benchmark_quantized_granite',
    Path(__file__).with_name('benchmark_quantized_granite.py'))
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_quality_metrics_report_divergence_without_exactness_gate():
    reference = np.array([[[2., 0.], [0., 2.]]])
    actual = np.array([[[0., 2.], [0., 2.]]])
    metric = BENCHMARK.compare_logits(reference, actual, [[0, 1]])
    assert metric['max_abs_logit_error'] == 2
    assert metric['top1_agreement'] == .5
    assert metric['mean_kl_reference_to_variant'] == pytest.approx(np.tanh(1.))
    assert metric['mean_continuation_nll_delta'] == pytest.approx(1.)
    equal = BENCHMARK.compare_logits(reference+10000, reference+10000, [[0, 1]])
    assert equal['mean_kl_reference_to_variant'] == 0
    assert equal['top1_agreement'] == 1


def test_metric_input_and_label_validation():
    for actual in (np.ones((3, 4)), np.full((1, 2), np.nan)):
        with pytest.raises(ValueError):
            BENCHMARK.compare_logits(np.zeros((1, 2)), actual)
    with pytest.raises(ValueError, match='labels'):
        BENCHMARK.compare_logits(np.zeros((1, 2)), np.zeros((1, 2)), [2])


def test_expected_sidecar_covers_legacy_experts_and_q8_attention_but_no_router():
    prefix = 'model.layers.0.block_sparse_moe.'
    cp = SimpleNamespace(tensors={prefix+'input_linear.weight': {'shape': [2, 64, 64]},
        prefix+'output_linear.weight': {'shape': [2, 64, 32]},
        prefix+'router.layer.weight': {'shape': [2, 64]},
        'model.layers.0.self_attn.q_proj.weight': {'shape': [64, 64]},
        'model.embed_tokens.weight': {'shape': [100, 64]}})
    experts = BENCHMARK.expected_entries(cp, 'experts')
    mixed = BENCHMARK.expected_entries(cp, 'mixed')
    assert len(experts) == 4 and all(value[3] == 4 for value in experts.values())
    assert len(mixed) == 5
    assert mixed['model.layers.0.self_attn.q_proj.weight'][3] == 8
    assert prefix+'router.layer.weight' not in mixed


def test_all_variants_keep_identical_memory_budgets(tmp_path):
    args = SimpleNamespace(device='cpu', cache_mib=0, pack_root=tmp_path)
    configs = [BENCHMARK.configuration(args, tmp_path/'metadata', variant) for variant in BENCHMARK.VARIANTS]
    for cfg in configs:
        assert cfg.cache_bytes == 0
        assert (cfg.io_bytes, cfg.kv_bytes, cfg.workspace_bytes, cfg.reserve_bytes) == (
            32*BENCHMARK.MiB, 32*BENCHMARK.MiB, 256*BENCHMARK.MiB, 1536*BENCHMARK.MiB)
    assert configs[0].quantized_dir is None
    assert Path(configs[1].quantized_dir).name == 'quantized-granite-expert4'


def test_eos_stopped_generation_counts_actual_tokens_and_length_divergence():
    baseline = [7, 8, 4, 5, 0]
    shorter = BENCHMARK.generation_summary([7, 8, 4, 0], 2, baseline, .5, [0])
    assert shorter['new_tokens'] == 2
    assert shorter['tokens_per_second'] == 4
    assert shorter['stop_reason'] == 'eos'
    assert shorter['first_continuation_divergence'] == 1
    assert shorter['continuation_length_difference'] == -1
    assert shorter['tokens_after_first_eos'] == 0
    prefix_only = BENCHMARK.generation_summary([7, 8, 4, 5], 2, baseline, 1., [0])
    assert prefix_only['first_continuation_divergence'] == 2
    assert prefix_only['stop_reason'] == 'max_new_tokens'
    with pytest.raises(ValueError, match='first EOS'):
        BENCHMARK.generation_summary([7, 8, 4, 0, 0], 2, baseline, .5, [0])


def test_variable_reference_length_preserves_teacher_forcing_alignment():
    # Prompt has two positions, reference emits a token followed by EOS.
    ids, baseline = [7, 8], [7, 8, 3, 0]
    reference_positions = len(baseline)-len(ids)
    all_logits = np.zeros((1, len(baseline)-1, 10))
    all_logits[0, 1, 3] = 5
    all_logits[0, 2, 0] = 5
    kept = all_logits[:, -reference_positions:]
    metrics = BENCHMARK.compare_logits(kept, kept, [baseline[len(ids):]])
    assert metrics['positions'] == 2
    assert metrics['reference_mean_continuation_nll'] < .1
