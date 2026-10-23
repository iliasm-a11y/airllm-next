"""Persistent draft-state reconciliation and real native Llama regression tests."""
from types import SimpleNamespace

import numpy as np
import pytest

from airllm.nextgen.incremental_draft import (
    IncrementalModelDraft, IncrementalReencodedModelDraft)
from airllm.nextgen.speculative import greedy_speculative_generate
from test_speculative import ToyCache, ToyTarget, expected


class NativeToy(ToyTarget):
    """Causal state oracle: it records every input actually put through KV."""
    def __init__(self, vocabulary=300):
        super().__init__(vocabulary)
        self.config = dict(vocab_size=vocabulary, max_position_embeddings=512)
        self.options = SimpleNamespace(max_context=512)

    def new_session(self):
        return ToyCache()


def proposals(model, prefix, count):
    return expected(model, prefix, count)[len(prefix):]


def test_accepted_round_reuses_kv_and_only_forwards_the_new_suffix():
    model = NativeToy()
    draft = IncrementalModelDraft(model)
    prefix = list(range(40))
    first = draft(prefix, 8)
    assert first == proposals(model, prefix, 8)
    calls = len(model.calls)
    accepted = prefix + first + [42]  # Target's accepted block plus bonus.
    second = draft(accepted, 4)
    assert second == proposals(model, accepted, 4)
    assert [len(row[0][0]) for row in model.calls[calls:]] == [2, 1, 1, 1]
    assert draft.stats['prefix_forwarded_tokens'] == 42
    assert draft.stats['forwarded_tokens'] == 52
    assert draft.stats['kv_reused_tokens'] == 47
    assert draft.stats['draft_tokens'] == 12
    assert draft.stats['draft_seconds'] > 0


@pytest.mark.parametrize('accepted', [0, 1, 2, 6])
def test_rejection_truncates_speculative_tail_and_replays_one_common_token(accepted):
    model = NativeToy()
    draft = IncrementalModelDraft(model)
    prefix = [1, 7, 2, 4]
    first = draft(prefix, 8)
    corrected = prefix + first[:accepted] + [(first[accepted] + 1) % model.vocabulary]
    calls = len(model.calls)
    assert draft(corrected, 3) == proposals(model, corrected, 3)
    assert len(model.calls[calls][0][0]) == 2
    assert draft._tokens == corrected + proposals(model, corrected, 3)[:-1]
    assert draft._cache.values[:draft._cache.position] == draft._tokens
    assert draft.stats['reconciliation_replayed_tokens'] == 1


def test_identical_prefix_count_one_reuses_cached_next_token_without_a_forward():
    model = NativeToy()
    draft = IncrementalModelDraft(model)
    first = draft([1, 2, 3], 1)
    calls = len(model.calls)
    assert draft([1, 2, 3], 1) == first
    assert len(model.calls) == calls


@pytest.mark.parametrize('prefix', [[1, 7], [9, 3, 8], [1, 7, 2, 4]])
def test_shorter_or_unrelated_prefix_cannot_reuse_stale_tail(prefix):
    model = NativeToy()
    draft = IncrementalModelDraft(model)
    draft([1, 7, 2, 4], 7)
    assert draft(prefix, 4) == proposals(model, prefix, 4)
    assert draft._cache.values[:draft._cache.position] == draft._tokens


def test_reset_releases_kv_without_closing_or_resetting_model_weights():
    model = NativeToy()
    draft = IncrementalModelDraft(model)
    draft([1, 2, 3], 3)
    before = draft.stats['forwarded_tokens']
    draft.reset()
    assert draft._cache is None and draft._tokens == []
    assert draft([1, 2], 1) == proposals(model, [1, 2], 1)
    assert draft.stats['forwarded_tokens'] == before + 2


def test_failure_after_cache_mutation_discards_all_draft_state():
    model = NativeToy()
    draft = IncrementalModelDraft(model)
    draft([1, 2, 3], 3)
    original = model.forward
    def fail_after_mutation(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError('simulated failed draft execution')
    model.forward = fail_after_mutation
    with pytest.raises(OSError):
        draft([1, 2, 8], 4)
    assert draft._cache is None and draft.stats['failed_calls'] == 1
    model.forward = original
    assert draft([1, 2, 8], 4) == proposals(model, [1, 2, 8], 4)


@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_invalid_count_fails_before_cache_work(value):
    model = NativeToy()
    draft = IncrementalModelDraft(model)
    with pytest.raises(ValueError):
        draft([1, 2], value)
    assert not model.calls


def test_zero_and_context_limit_return_no_proposals_without_work():
    model = NativeToy()
    model.options.max_context = 3
    draft = IncrementalModelDraft(model)
    assert draft([1, 2], 0) == []
    assert draft([1, 2, 3], 4) == []
    assert not model.calls
    assert len(draft([1, 2], 4)) == 1


@pytest.mark.parametrize('eos', [0, [0], [0, 2]])
def test_default_eos_including_zero_and_multiple_ids_stops_before_forwarding_eos(eos):
    model = NativeToy()
    model.config['eos_token_id'] = eos
    model.next = lambda prefix: 0
    draft = IncrementalModelDraft(model)
    assert draft([1, 2, 3], 8) == [0]
    assert len(model.calls) == 1
    assert draft._cache.position == 3
    assert draft.stats['eos_proposals'] == 1


def test_explicit_empty_eos_list_allows_candidate_continuation():
    model = NativeToy()
    model.config['eos_token_id'] = [0, 2]
    model.next = lambda prefix: 0
    assert IncrementalModelDraft(model, eos_token_ids=[])([1, 2], 4) == [0] * 4


def test_target_verification_stays_exact_with_a_persistently_wrong_cached_draft():
    target, model = ToyTarget(), NativeToy(vocabulary=17)
    model.next = lambda prefix: (target.next(prefix) + 1) % target.vocabulary
    draft = IncrementalModelDraft(model)
    result = greedy_speculative_generate(target, draft, [[1, 8, 4]], 40,
        adaptive=False, max_depth=8)
    assert result.tokens == expected(target, [1, 8, 4], 40)
    assert draft.stats['rollback_tokens'] > 0
    assert draft.stats['prefix_forwarded_tokens'] < 3 * 40


class CharacterTokenizer:
    def __init__(self, offset=0):
        self.offset = offset
    def encode(self, text, **kwargs):
        return [ord(c) + self.offset for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(chr(i - self.offset) for i in ids)
    def get_vocab(self):
        return {chr(i): i + self.offset for i in range(200)}


def character_model(tokenizer):
    model = NativeToy(vocabulary=400)
    model.tokenizer = tokenizer
    model.next = lambda prefix: prefix[-1] + 1
    return model


def test_same_tokenizer_validation_rejects_different_token_mappings():
    model = character_model(CharacterTokenizer(10))
    with pytest.raises(ValueError, match='token mappings differ'):
        IncrementalModelDraft(model, CharacterTokenizer())
    IncrementalModelDraft(model, CharacterTokenizer(10))


def test_reencoded_proposals_map_ids_and_reuse_accepted_prefix():
    target = CharacterTokenizer()
    model = character_model(CharacterTokenizer(10))
    draft = IncrementalReencodedModelDraft(model, target)
    assert draft(target.encode('abc'), 4) == target.encode('defg')
    assert draft(target.encode('abcdefg'), 2) == target.encode('hi')
    assert draft.stats['prefix_forwarded_tokens'] == 4
    assert draft.stats['target_candidates'] == 6
    assert draft.stats['reencoding_seconds'] > 0
    assert draft.stats['proposal_seconds'] >= draft.stats['draft_seconds']


class MergeABTokenizer(CharacterTokenizer):
    def encode(self, text, **kwargs):
        return [300 if item == 'ab' else ord(item) for item in _split_ab(text)]
    def decode(self, ids, **kwargs):
        return ''.join('ab' if item == 300 else chr(item) for item in ids)


def _split_ab(text):
    parts = []
    while text:
        size = 2 if text.startswith('ab') else 1
        parts.append(text[:size])
        text = text[size:]
    return parts


def test_draft_retokenization_reconciles_changed_token_boundary():
    target = CharacterTokenizer()
    model = character_model(MergeABTokenizer())
    model.next = lambda prefix: ord('b') if prefix[-1] == ord('a') else ord('c')
    draft = IncrementalReencodedModelDraft(model, target)
    assert draft(target.encode('a'), 3) == target.encode('bcc')
    # The accepted string 'ab' is one draft token, replacing the cached 'a'.
    assert draft(target.encode('ab'), 2) == target.encode('cc')
    assert draft.stats['rollback_tokens'] == 3
    assert draft.incremental._tokens[0] == 300


def test_unstable_target_seam_returns_nothing_before_target_verification():
    target = MergeABTokenizer()
    model = character_model(CharacterTokenizer())
    draft = IncrementalReencodedModelDraft(model, target)
    assert draft([ord('a')], 2) == []
    assert draft.stats['unstable_seam'] == 1


def test_native_llama_cached_proposals_match_full_greedy_after_rejection(tmp_path):
    from conftest import save_tiny
    from test_native import options
    from airllm.nextgen.llama_runtime import AirLLMNextLlamaModel
    root = tmp_path / 'llama'
    save_tiny(root)
    with AirLLMNextLlamaModel(root, next_config=options(tmp_path)) as model:
        draft = IncrementalModelDraft(model)
        prefix = [1, 7, 8, 9]
        first = draft(prefix, 8)
        assert first == model.generate([prefix], max_new_tokens=8, eos_token_id=[]).tolist()[0][len(prefix):]
        corrected = prefix + first[:2] + [(first[2] + 1) % model.config['vocab_size']]
        result = draft(corrected, 8)
        baseline = model.generate([corrected], max_new_tokens=8, eos_token_id=[]).tolist()[0]
        assert result == baseline[len(corrected):]
        assert draft._cache.position == len(corrected) + 7
        assert draft.stats['reconciliation_replayed_tokens'] == 1
        assert draft.stats['kv_peak_bytes'] <= model.options.kv_bytes
