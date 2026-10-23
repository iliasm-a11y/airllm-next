"""Opt-in local greedy drafts retaining their own append-only native KV.

The draft cache is independent of the target cache. Each proposal reconciles
the supplied, target-accepted prefix before reusing any state. A rejected tail
is truncated, and at most one common-prefix token is replayed to recover the
next-token logits. No candidate bypasses the target's normal verification.
"""
from collections import Counter
from threading import RLock
from time import perf_counter

import numpy as np

from .generation import eos_ids, token_batch


class IncrementalModelDraft:
    """Adapt a loaded native model with persistent, bounded, truncatable KV.

    Token ids must have the same meaning as the target's. Supply both local
    tokenizers to check their vocabulary mappings. For different tokenizers,
    use :class:`IncrementalReencodedModelDraft` instead. This adapter owns one
    generation state; call ``reset()`` to release it between independent runs.
    The underlying model and its weight cache remain caller-owned.
    """

    def __init__(self, model, target_tokenizer=None, *, eos_token_ids=None):
        if not callable(getattr(model, 'forward', None)) or not callable(getattr(model, 'new_session', None)):
            raise TypeError('Incremental draft requires native forward() and new_session()')
        if target_tokenizer is not None:
            tokenizer = getattr(model, 'tokenizer', None)
            if tokenizer is None or tokenizer.get_vocab() != target_tokenizer.get_vocab():
                raise ValueError('Draft and target token mappings differ; use text re-encoding')
        self.model = model
        self.eos_token_ids = eos_ids(model.config.get('eos_token_id')
            if eos_token_ids is None else eos_token_ids, model.config['vocab_size'])
        self.checkpoint = getattr(model, 'checkpoint', None)
        self.stats = Counter()
        self._lock = RLock()
        self._cache = None
        self._tokens = []
        self._next_token = None

    def reset(self):
        """Release only this adapter's KV state; preserve measured counters."""
        with self._lock:
            self._cache = None
            self._tokens = []
            self._next_token = None
            self.stats['resets'] += 1

    def _forward(self, tokens, *, prefix):
        output = self.model.forward(np.asarray([tokens], dtype=np.int32),
            past_key_values=self._cache, logits_to_keep=1)
        cache = output.past_key_values
        if (cache is None or not callable(getattr(cache, 'truncate', None))
                or cache.position != len(self._tokens) + len(tokens)):
            raise ValueError('Incremental draft requires append-only truncatable KV')
        mx = getattr(self.model, 'mx', None)
        if mx is not None:
            with mx.stream(self.model.stream):
                token = int(mx.argmax(output.logits[:, -1, :], axis=-1).item())
        else:
            token = int(np.argmax(np.asarray(output.logits)[:, -1, :], axis=-1).item())
        self._cache = cache
        self._tokens.extend(tokens)
        self._next_token = token
        self.stats['forward_calls'] += 1
        self.stats['forwarded_tokens'] += len(tokens)
        self.stats['prefix_forwarded_tokens' if prefix else 'proposal_forwarded_tokens'] += len(tokens)

    def _reconcile(self, prefix):
        common = 0
        for old, new in zip(self._tokens, prefix):
            if old != new:
                break
            common += 1
        if common == len(prefix) == len(self._tokens) and self._next_token is not None:
            self.stats['kv_reused_tokens'] += common
            return
        if common < len(self._tokens):
            # We retain only one next-token result, not a vocabulary-sized
            # logits history. Replay the last common token after truncating.
            retained = max(0, common - 1)
            self.stats['rollback_tokens'] += len(self._tokens) - retained
            self.stats['reconciliation_replayed_tokens'] += common - retained
            self._cache.truncate(retained)
            del self._tokens[retained:]
        self.stats['kv_reused_tokens'] += len(self._tokens)
        self._forward(prefix[len(self._tokens):], prefix=True)

    def __call__(self, prefix, count):
        begin = perf_counter()
        try:
            return self._propose(prefix, count)
        finally:
            self.stats['draft_seconds'] += perf_counter() - begin

    def _propose(self, prefix, count):
        if type(count) is not int or count < 0:
            raise ValueError('Proposal length must be a nonnegative integer')
        if not count:
            return []
        prefix = token_batch([prefix], self.model.config['vocab_size']).tolist()[0]
        limit = min(self.model.options.max_context, self.model.config['max_position_embeddings'])
        count = min(count, limit - len(prefix))
        if count < 1:
            self.stats['context_limit'] += 1
            return []
        with self._lock:
            try:
                self._reconcile(prefix)
                result = []
                for index in range(count):
                    token = self._next_token
                    result.append(token)
                    if token in self.eos_token_ids:
                        self.stats['eos_proposals'] += 1
                        break
                    # The final candidate need not be put into KV until the
                    # next call confirms it belongs to the accepted prefix.
                    if index + 1 < count:
                        self._forward([token], prefix=False)
                self.stats['calls'] += 1
                self.stats['draft_tokens'] += len(result)
                self.stats['kv_peak_bytes'] = max(self.stats['kv_peak_bytes'], self._cache.peak_bytes)
                return result
            except BaseException:
                # Native forward is transactional, but unknown user wrappers
                # may fail after mutating state. Rebuild rather than reuse it.
                self.reset()
                self.stats['failed_calls'] += 1
                raise


class IncrementalReencodedModelDraft:
    """Conservative text conversion backed by a persistent native draft KV.

    Re-encoding the actual target prefix may change its last draft tokens.
    The incremental adapter compares token ids and rolls back that seam, just
    as it does a rejected candidate. Prefix and output-boundary stability are
    checked in the target tokenizer before returning any proposals.
    """

    def __init__(self, model, target_tokenizer, *, max_prefix_tokens=2048, eos_token_ids=None):
        if type(max_prefix_tokens) is not int or max_prefix_tokens < 1:
            raise ValueError('max_prefix_tokens must be a positive integer')
        if getattr(model, 'tokenizer', None) is None:
            raise ValueError('Draft requires a local tokenizer')
        self.model, self.target_tokenizer = model, target_tokenizer
        self.incremental = IncrementalModelDraft(model, eos_token_ids=eos_token_ids)
        self.checkpoint = self.incremental.checkpoint
        self.max_prefix_tokens = max_prefix_tokens
        self.stats = self.incremental.stats

    def reset(self):
        self.incremental.reset()

    def __call__(self, prefix, count):
        begin = perf_counter()
        native_before = self.stats['draft_seconds']
        try:
            return self._propose(prefix, count)
        finally:
            elapsed = perf_counter() - begin
            self.stats['proposal_seconds'] += elapsed
            self.stats['reencoding_seconds'] += max(0., elapsed - (self.stats['draft_seconds'] - native_before))

    def _propose(self, prefix, count):
        if type(count) is not int or count < 0:
            raise ValueError('Proposal length must be a nonnegative integer')
        if not count:
            return []
        prefix = list(prefix)
        text = self.target_tokenizer.decode(prefix, skip_special_tokens=False)
        if self.target_tokenizer.encode(text, add_special_tokens=False) != prefix:
            self.stats['unstable_prefix'] += 1
            return []
        ids = self.model.tokenizer.encode(text, add_special_tokens=False)
        if not ids or len(ids) > self.max_prefix_tokens:
            self.stats['prefix_limit'] += 1
            return []
        output = self.incremental(ids, count)
        if not output:
            return []
        continuation = self.model.tokenizer.decode(output, skip_special_tokens=True)
        retokenized = self.target_tokenizer.encode(text + continuation, add_special_tokens=False)
        self.stats['reencoding_calls'] += 1
        if retokenized[:len(prefix)] != prefix:
            self.stats['unstable_seam'] += 1
            return []
        result = retokenized[len(prefix):len(prefix) + count]
        self.stats['target_candidates'] += len(result)
        return result
