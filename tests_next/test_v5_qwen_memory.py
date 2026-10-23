"""Qwen transactional rollback must reserve replacement overlap during prefill."""
from types import SimpleNamespace

import numpy as np
import pytest

from airllm.nextgen.qwen_runtime import QwenSession


def test_second_prefill_bank_growth_rejects_peak_and_restores_entry_snapshot():
    owner = SimpleNamespace(options=SimpleNamespace(kv_bytes=100, max_context=128),
                            config={'max_position_embeddings': 128})
    cache = QwenSession(owner)
    original = np.arange(4, dtype=np.float32)  # 16 bytes at the committed entry.
    cache.store('kv', 0, original)
    cache.position = 1
    entry = cache.snapshot()
    first_chunk = np.zeros(10, dtype=np.float32)  # 40-byte current bank.
    cache.store('kv', 0, first_chunk)
    cache.position = 2
    assert cache.bytes + cache.retained == 56
    # A 64-byte replacement would fit the 80-byte eventual steady state, but
    # current40 + retained16 + replacement64 =120 exceeds the 100-byte cap.
    with pytest.raises(MemoryError, match='rollback states'):
        cache.reserve('kv', 0, 64)
    assert cache.banks['kv'][0] is first_chunk
    assert cache.position == 2 and entry.active
    cache.restore(entry)
    assert cache.banks['kv'][0] is original
    assert cache.position == 1 and cache.retained == 0 and not entry.active


@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_invalid_qwen_replacement_reservations_fail_before_mutation(value):
    owner = SimpleNamespace(options=SimpleNamespace(kv_bytes=100, max_context=128),
                            config={'max_position_embeddings': 128})
    cache = QwenSession(owner)
    with pytest.raises(ValueError, match='nonnegative integer'):
        cache.reserve('recurrent', 0, value)
    assert cache.bytes == cache.retained == cache.position == 0
