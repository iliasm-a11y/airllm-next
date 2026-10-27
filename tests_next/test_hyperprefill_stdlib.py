"""Planner admission checks without importing MLX, Torch or test conftest."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location('isolated_hyperprefill',
    Path(__file__).resolve().parents[1] / 'air_llm/airllm/nextgen/hyperprefill.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
HyperPrefillPolicy = MODULE.HyperPrefillPolicy


def config():
    return dict(hidden_size=2048, num_experts_per_tok=8, moe_intermediate_size=512,
                num_attention_heads=16, num_key_value_heads=2, head_dim=256,
                linear_num_key_heads=16, linear_num_value_heads=32,
                linear_key_head_dim=128, linear_value_head_dim=128,
                linear_conv_kernel_dim=4, layer_types=['linear_attention'] * 30 + ['full_attention'] * 10,
                max_position_embeddings=4096)


def options(**changes):
    values = dict(workspace_bytes=128 << 20, kv_bytes=256 << 20,
                  reserve_bytes=3 << 30, max_context=1280)
    values.update(changes)
    return SimpleNamespace(**values)


class HyperPrefillStdlibTests(unittest.TestCase):
    def choose(self, policy=None, settings=None, **changes):
        inputs = dict(cache_position=0, remaining_tokens=364, available_bytes=6 << 30)
        inputs.update(changes)
        return (policy or HyperPrefillPolicy()).select(config(), settings or options(), **inputs)

    def test_cap_reuses_more_tokens_and_preserves_all_memory_limits(self):
        settings = options()
        before = deepcopy(vars(settings))
        decision = self.choose(settings=settings)
        self.assertEqual(decision.chunk_tokens, 128)
        self.assertLessEqual(decision.estimated_workspace_bytes, settings.workspace_bytes // 2)
        self.assertLessEqual(decision.estimated_context_bytes, settings.kv_bytes)
        self.assertEqual(vars(settings), before)

    def test_constrained_workspace_selects_largest_admissible_chunk(self):
        settings = options(workspace_bytes=16 << 20)
        decision = self.choose(settings=settings)
        self.assertGreaterEqual(decision.chunk_tokens, 1)
        self.assertLess(decision.chunk_tokens, 128)
        self.assertEqual(decision.reason, 'memory_budget')
        next_workspace = HyperPrefillPolicy.estimates(config(), 0, decision.chunk_tokens + 1)[0]
        self.assertGreater(next_workspace, settings.workspace_bytes // 2)

    def test_reserve_and_kv_fail_before_a_single_token_is_admitted(self):
        for changes in ({'available_bytes': 3 << 30}, {'retained_bytes': 256 << 20}):
            with self.subTest(changes=changes), self.assertRaises(MemoryError):
                self.choose(**changes)
        with self.assertRaises(MemoryError):
            self.choose(settings=options(kv_bytes=1))

    def test_long_context_and_decode_respect_remaining_positions(self):
        decision = self.choose(cache_position=1278, remaining_tokens=10)
        self.assertEqual(decision.chunk_tokens, 2)
        decode = self.choose(cache_position=1200, remaining_tokens=1)
        self.assertEqual((decode.chunk_tokens, decode.reason), (1, 'decode'))
        with self.assertRaises(MemoryError):
            self.choose(cache_position=1280, remaining_tokens=1)

    def test_invalid_caps_and_request_dimensions_are_rejected(self):
        for value in (True, False, 0, -1, 1.5, '128'):
            with self.subTest(cap=value), self.assertRaises(ValueError):
                HyperPrefillPolicy(value)
        for key, value in (('cache_position', -1), ('remaining_tokens', 0), ('batch_size', True),
                           ('available_bytes', 1.5), ('cache_payload_bytes', -1), ('retained_bytes', -1),
                           ('dtype_bytes', 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.choose(**{key: value})

    def test_small_known_dimensions_have_hand_calculated_memory_charge(self):
        dimensions = dict(hidden_size=4, num_experts_per_tok=2, moe_intermediate_size=3,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=2,
                          linear_num_key_heads=1, linear_num_value_heads=2,
                          linear_key_head_dim=2, linear_value_head_dim=2,
                          linear_conv_kernel_dim=4, layer_types=['full_attention'])
        # Four new tokens at position two: residuals128, routed640,
        # attention768, one full-layer KV96, transient conv128 bytes.
        self.assertEqual(HyperPrefillPolicy.estimates(dimensions, 2, 4), (896, 96, 128))


if __name__ == '__main__':
    unittest.main()
