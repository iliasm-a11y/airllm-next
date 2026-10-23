"""Admission plans reject infeasible states without loading tensor payloads."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from airllm.nextgen.config import NextConfig, GiB
from airllm.nextgen.preflight import preflight_checkpoint


def llama_config():
    return dict(model_type='llama', num_hidden_layers=3, num_attention_heads=8,
                num_key_value_heads=2, hidden_size=128, head_dim=16,
                vocab_size=96, max_position_embeddings=8192)


def qwen_config():
    return dict(model_type='qwen3_5_moe_text', num_hidden_layers=92,
                full_attention_interval=4, linear_num_key_heads=16,
                linear_num_value_heads=128, linear_key_head_dim=128,
                linear_value_head_dim=128, linear_conv_kernel_dim=4,
                mamba_ssm_dtype='float32', num_key_value_heads=4,
                head_dim=256, dtype='bfloat16', max_position_embeddings=262144)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write_model(llama_config())

    def write_model(self, config, *, shard='model-00001.safetensors', payload=1000,
                    present=True):
        (self.root / 'config.json').write_text(json.dumps(config))
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({
            'metadata': {'total_size': payload}, 'weight_map': {'weight': shard}}))
        if present and not Path(shard).is_absolute() and '..' not in Path(shard).parts:
            (self.root / shard).write_bytes(b'not a validated safetensors file')

    def report(self, **options):
        defaults = dict(tokens=10, available_memory_bytes=16 * GiB,
                        storage_free_bytes=2_000_000_000_000)
        defaults.update(options)
        return preflight_checkpoint(self.root, **defaults)

    def test_presence_pass_never_promotes_integrity_or_generation(self):
        before = sorted(p.name for p in self.root.iterdir())
        report = self.report()
        self.assertTrue(report['preflight_passed'])
        self.assertFalse(report['generation_validated'])
        self.assertFalse(report['files']['tensor_headers_checked'])
        self.assertFalse(report['files']['payload_integrity_verified'])
        self.assertFalse(report['downloads_performed'])
        self.assertFalse(report['writes_performed'])
        self.assertFalse(report['memory']['is_complete_peak_bound'])
        self.assertEqual(before, sorted(p.name for p in self.root.iterdir()))

    def test_missing_giant_shards_block_download_on_small_disk(self):
        (self.root / 'model-00001.safetensors').unlink()
        self.write_model(llama_config(), payload=755617140416, present=False)
        report = self.report(storage_free_bytes=30 * GiB)
        self.assertFalse(report['preflight_passed'])
        self.assertFalse(report['files']['all_declared_files_present'])
        self.assertEqual(report['storage']['download_gate'],
                         'blocked_fresh_payload_exceeds_free_space_with_reserve')

    def test_download_is_not_approved_from_metadata_fit_alone(self):
        (self.root / 'model-00001.safetensors').unlink()
        report = self.report()
        self.assertTrue(report['storage']['fresh_declared_payload_fits_with_reserve'])
        self.assertEqual(report['storage']['download_gate'],
                         'metadata_only_prerequisites_unverified')

    def test_empty_shard_is_not_present_checkpoint(self):
        (self.root / 'model-00001.safetensors').write_bytes(b'')
        report = self.report()
        self.assertFalse(report['preflight_passed'])
        self.assertEqual(report['files']['empty_files'], ['model-00001.safetensors'])

    def test_traversal_and_symlink_do_not_escape_root(self):
        for shard in ('../outside.safetensors', '/outside.safetensors'):
            self.write_model(llama_config(), shard=shard, present=False)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                self.report()
        (self.root / 'escape.safetensors').symlink_to(self.root.parent / 'outside')
        self.write_model(llama_config(), shard='escape.safetensors', present=False)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.report()

    def test_qwen_includes_full_transaction_snapshot(self):
        self.write_model(qwen_config())
        cfg = replace(NextConfig(), kv_bytes=2 * GiB, max_context=8192)
        report = self.report(config=cfg, tokens=4096, allow_experimental=True)
        state = report['state']
        self.assertEqual(state['recurrent_bytes'], 578813952)
        self.assertEqual(state['kv_bytes'], 385875968)
        self.assertEqual(state['retained_transaction_snapshot_bytes'], state['payload_bytes'])
        self.assertEqual(state['bank_replacement_reservation_bytes'], 16777216)
        self.assertEqual(state['required_state_reservation_bytes'], 2 * state['payload_bytes'] + 16777216)
        self.assertTrue(report['preflight_passed'])
        self.assertFalse(self.report(tokens=4096, allow_experimental=True)['preflight_passed'])

    def test_qwen_experimental_and_speculation_are_distinct_gates(self):
        self.write_model(qwen_config())
        report = self.report(tokens=4)
        self.assertFalse(report['preflight_passed'])
        cfg = replace(NextConfig(), kv_bytes=2 * GiB)
        report = self.report(config=cfg, tokens=4, allow_experimental=True, speculative_depth=4)
        self.assertIn('Qwen recurrent speculative verification is not enabled', report['blockers'])

    def test_nvfp4_is_not_mislabelled_as_affine_int4(self):
        self.write_model(qwen_config() | {'quantization_config': {'quant_method': 'modelopt'}})
        report = self.report(tokens=0, allow_experimental=True)
        self.assertFalse(report['preflight_passed'])
        self.assertFalse(report['inspection']['quantization']['direct_next_read'])

    def test_agent_registration_does_not_duplicate_weights(self):
        one = self.report()
        many = self.report(logical_agents=300)
        self.assertEqual(one['memory']['configured_total_with_reserve_bytes'],
                         many['memory']['configured_total_with_reserve_bytes'])
        two = self.report(logical_agents=300, concurrent_inferences=2)
        cfg = NextConfig()
        self.assertEqual(two['memory']['configured_total_with_reserve_bytes'] -
                         one['memory']['configured_total_with_reserve_bytes'],
                         cfg.io_bytes + cfg.kv_bytes + cfg.workspace_bytes)

    def test_near_pressure_rejects_before_load(self):
        report = self.report(available_memory_bytes=GiB)
        self.assertIn('Configured inference budgets plus reserve exceed available memory', report['blockers'])

    def test_speculative_tokens_and_float32_state_count_against_caps(self):
        cfg = replace(NextConfig(), max_context=16, dtype='float32')
        report = self.report(config=cfg, tokens=15, speculative_depth=2)
        self.assertIn('Requested prefix and proposals exceed configured max_context', report['blockers'])
        self.assertEqual(report['state']['kv_bytes'], 17 * 3 * 2 * 2 * 16 * 4)

    def test_single_file_with_unknown_payload_still_requires_local_file(self):
        (self.root / 'model.safetensors.index.json').unlink()
        (self.root / 'model-00001.safetensors').unlink()
        report = self.report()
        self.assertEqual(report['storage']['download_gate'], 'blocked_unknown_payload_size')
        self.assertEqual(report['files']['missing_files'], ['model.safetensors'])

    def test_unindexed_local_shards_are_inventory_not_integrity(self):
        (self.root / 'model.safetensors.index.json').unlink()
        report = self.report()
        self.assertTrue(report['files']['all_declared_files_present'])
        self.assertFalse(report['files']['payload_integrity_verified'])
        self.assertEqual(report['storage']['download_gate'], 'not_needed_for_declared_files')

    def test_batch_capacity_estimate_does_not_enable_batch_execution(self):
        report = self.report(batch_size=2)
        self.assertIn('Next runtimes support unpadded batch_size=1 only', report['blockers'])
        self.assertFalse(report['preflight_passed'])

    def test_kv_reservation_counts_capacity_rounding_and_growth_overlap(self):
        first = self.report(tokens=1)
        self.assertEqual(first['state']['allocated_capacity_tokens'], 128)
        self.assertEqual(first['state']['kv_growth_overlap_reservation_bytes'], 0)
        grown = self.report(tokens=129)
        self.assertEqual(grown['state']['allocated_capacity_tokens'], 256)
        self.assertEqual(grown['state']['kv_growth_overlap_reservation_bytes'], 128 * 2 * 2 * 16 * 2)
        cfg = replace(NextConfig(), kv_bytes=grown['state']['payload_bytes'])
        blocked = self.report(config=cfg, tokens=129)
        self.assertIn('State plus retained rollback reservation exceeds kv_bytes', blocked['blockers'])

    def test_invalid_resource_values_fail_early(self):
        for kwargs in ({'tokens': True}, {'speculative_depth': -1}, {'batch_size': 0},
                       {'available_memory_bytes': -1}, {'storage_free_bytes': True},
                       {'logical_agents': 1, 'concurrent_inferences': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.report(**kwargs)


if __name__ == '__main__':
    unittest.main()
