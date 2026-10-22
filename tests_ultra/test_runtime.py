import json
import pytest
import torch
from airllm.ultra import UltraConfig, AirLLMUltraModel
from conftest import save_tiny


def options(tmp_path, **kwargs):
    defaults = dict(cache_bytes=2**20, prefetch_bytes=2**20, reserve_bytes=0,
                    max_group_bytes=4**12, cache_dir=str(tmp_path / 'meta'), load_tokenizer=False)
    defaults.update(kwargs)
    return UltraConfig(**defaults)


def test_logits_decode_and_kv_match_reference(tiny, tmp_path):
    path, ref = tiny
    ids = torch.tensor([[1, 8, 9, 10]])
    with AirLLMUltraModel(path, device='cpu', dtype=torch.float32,
                         ultra=options(tmp_path)) as model, torch.inference_mode():
        expected = ref(ids, use_cache=True)
        actual = model(ids, use_cache=True)
        torch.testing.assert_close(actual.logits, expected.logits, atol=3e-6, rtol=3e-5)
        next_id = actual.logits[:, -1].argmax(-1)[:, None]
        expected2 = ref(next_id, use_cache=True, past_key_values=expected.past_key_values)
        actual2 = model(next_id, use_cache=True, past_key_values=actual.past_key_values)
        torch.testing.assert_close(actual2.logits, expected2.logits, atol=3e-6, rtol=3e-5)
        expected_ids = ref.generate(ids, max_new_tokens=4, do_sample=False)
        actual_ids = model.generate(ids, max_new_tokens=4, do_sample=False)
        assert torch.equal(actual_ids, expected_ids)
        assert all(p.device.type == 'meta' for p in model.model.parameters())
        assert model.cache.peak_bytes <= model.ultra.cache_bytes
        assert model.prefetch.peak_bytes <= model.ultra.prefetch_bytes


def test_tied_embeddings_stream_instead_of_pinning_table(tmp_path):
    root = tmp_path / 'tied'
    ref = save_tiny(root, tied=True)
    ids = torch.tensor([[1, 9]])
    with AirLLMUltraModel(root, device='cpu', dtype=torch.float32, ultra=options(tmp_path)) as model:
        with torch.inference_mode():
            torch.testing.assert_close(model(ids).logits, ref(ids).logits)
        assert all(p.device.type == 'meta' for p in model.model.parameters())


def test_no_split_and_cached_second_request(tiny, tmp_path):
    root, ref = tiny
    with AirLLMUltraModel(root, device='cpu', dtype=torch.float32,
                         ultra=options(tmp_path, cache_bytes=16*2**20, prefetch=False)) as model:
        ids = torch.tensor([[1]])
        model(ids)
        reads = model.checkpoint.stats['tensor_reads']
        model(ids)
        assert model.checkpoint.stats['tensor_reads'] == reads
        assert model.cache.stats['hits'] > 0
        assert not (root / 'splitted_model').exists()
    with pytest.raises(RuntimeError, match='closed'):
        model(ids)


def test_reopen_reuses_valid_manifest_and_routes(tiny, tmp_path):
    root, _ = tiny
    cfg = options(tmp_path)
    with AirLLMUltraModel(root, device='cpu', ultra=cfg) as model:
        model(torch.tensor([[1, 8]]))
        assert model.checkpoint.stats['manifest_hit'] == 0
    with AirLLMUltraModel(root, device='cpu', ultra=cfg) as model:
        assert model.checkpoint.stats['manifest_hit'] == 1
        assert model.checkpoint.stats['header_bytes'] == 0
        if model.expert_sources:
            assert model.routes


def test_memory_guard_rejects_before_materialization(tmp_path):
    root = tmp_path / 'small'
    save_tiny(root)
    with pytest.raises(MemoryError, match='max_group_bytes'):
        AirLLMUltraModel(root, device='cpu', ultra=options(tmp_path, max_group_bytes=100))


def test_auto_entry_point(tmp_path):
    from airllm import AutoModel
    root = tmp_path / 'entry'
    save_tiny(root)
    with AutoModel.from_pretrained(root, device='cpu', ultra=options(tmp_path)) as model:
        assert model(torch.tensor([[1]])).logits.shape[-1] == 96


def test_only_routed_expert_slices_are_read(tmp_path):
    root = tmp_path / 'experts'
    save_tiny(root, 'mixtral')
    with AirLLMUltraModel(root, device='cpu', dtype=torch.float32,
                         ultra=options(tmp_path, prefetch=False, cache_bytes=0)) as model:
        bank = next(m for m in model.model.modules() if m.__class__.__name__ == 'DiskExperts')
        before = model.checkpoint.stats['logical_bytes']
        bank(torch.randn(1, 48), torch.tensor([[2, 5]]), torch.tensor([[0.4, 0.6]]))
        delta = model.checkpoint.stats['logical_bytes'] - before
        assert delta == 2 * bank.source.size  # f32 checkpoint/runtime
        assert delta < model.checkpoint.nbytes(model.checkpoint.keys(bank.source.storage_prefix))
