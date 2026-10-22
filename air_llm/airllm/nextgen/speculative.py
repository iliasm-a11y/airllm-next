"""Exact greedy draft-and-verify decoding with append-only KV rollback.

Every emitted token is the target's argmax under its accepted prefix. A draft
only proposes inputs for one causal target forward; it never supplies an
unverified answer. Sampling and tree verification are intentionally unsupported.
The draft is an explicitly supplied local model/callable; no download or service
lookup occurs here. Block evaluation can have ordinary floating-point rounding
differences from one-token evaluation, as with any batched inference.
"""
from dataclasses import dataclass
from itertools import islice
from time import perf_counter
import numbers
import numpy as np
from .generation import token_batch,eos_ids,nonnegative_integer


@dataclass
class SpeculativeResult:
    tokens: list[int]
    cache: object
    metrics: dict


class LocalModelDraft:
    """Adapt an already loaded local model's greedy generate method.

    This conservative adapter recomputes the small draft's prefix each round.
    A caller can instead provide propose(prefix, count) with its own bounded,
    correctly managed draft state. The target cache is never shared with it.
    """
    def __init__(self, model, eos_token_ids=()):
        self.model = model
        self.eos_token_ids = list(eos_token_ids)

    def __call__(self, prefix, count):
        generated = self.model.generate(
            np.asarray([prefix], dtype=np.int32), max_new_tokens=count,
            temperature=0., eos_token_id=self.eos_token_ids)
        values = generated.tolist()[0]
        if values[:len(prefix)] != prefix:
            raise ValueError('Draft generate must preserve the supplied prefix')
        return values[len(prefix):]


@dataclass
class AdaptiveSpeculation:
    """A measured wall-time guard, not a prediction of SSD or GPU throughput.

    The observed single-token target pass supplies a baseline. After warmup,
    repeated slower rounds disable speculation for this generation. Depth is
    reduced on low acceptance, and increased only after fully accepted blocks.
    A subsequent generation calibrates again because cache/thermal state changes.
    """
    max_depth: int = 4
    enabled: bool = True
    warmup_rounds: int = 2
    slow_rounds_to_disable: int = 2
    slowdown_tolerance: float = 1.05

    def __post_init__(self):
        if not isinstance(self.max_depth, int) or isinstance(self.max_depth, bool) or self.max_depth < 1:
            raise ValueError('Speculative depth must be a positive integer')
        self.depth = min(2, self.max_depth) if self.enabled else self.max_depth
        self.active = True
        self.rounds = 0
        self.slow_streak = 0
        self.zero_accept_streak = 0
        self.disabled_reason = None

    def observe(self, *, seconds, emitted, proposed, accepted, baseline_seconds):
        self.rounds += 1
        if not self.enabled:
            return
        self.zero_accept_streak = self.zero_accept_streak+1 if proposed and accepted==0 else 0
        if accepted == proposed and proposed:
            self.depth = min(self.max_depth, self.depth + 1)
        elif proposed and accepted * 2 < proposed:
            self.depth = max(1, self.depth // 2)
        # Repeatedly accepting no draft tokens cannot amortize any target pass.
        # This observed progress guard remains valid even when a noisy initial
        # decode timing badly overestimates the wall-time fallback baseline.
        if self.zero_accept_streak >= self.warmup_rounds+self.slow_rounds_to_disable:
            self.active = False
            self.disabled_reason = 'no_accepted_draft_tokens'
            return
        if self.rounds <= self.warmup_rounds or baseline_seconds is None:
            return
        slow = seconds / max(1, emitted) > baseline_seconds * self.slowdown_tolerance
        self.slow_streak = self.slow_streak + 1 if slow else 0
        if self.slow_streak >= self.slow_rounds_to_disable:
            self.active = False
            self.disabled_reason = 'measured_draft_plus_verify_cost_exceeds_decode_baseline'


def _host_argmax(logits):
    return np.argmax(np.asarray(logits), axis=-1).tolist()


def greedy_speculative_generate(target, propose, input_ids, max_new_tokens,
                                *, eos_token_ids=(), max_depth=4, adaptive=True,
                                argmax_tokens=None, bytes_read=None,
                                clock=perf_counter,vocab_size=None,encoded_bytes_read=None):
    """Generate using causal block verification and a lagging pending token.

    Cache invariant: after each round it contains exactly the committed output
    prefix *except* its final pending token. On rejection, truncate before the
    rejected draft token and append the target correction on the next forward.
    This avoids an extra weight pass just to process a correction/bonus token.

    target.forward must return logits for every supplied position and an
    append-only cache with position and truncate(position). propose returns new
    token ids only. The target and draft must use the same tokenizer/vocabulary;
    callers are responsible for establishing this semantic compatibility.
    """
    max_new_tokens = nonnegative_integer(max_new_tokens, 'max_new_tokens')
    ids = token_batch(input_ids, vocab_size)
    tokens = ids[0].astype(np.int64).tolist()
    prefix_length = len(tokens)
    eos = eos_ids(eos_token_ids, vocab_size)
    choose = argmax_tokens or _host_argmax
    read = bytes_read or (lambda: 0)
    policy = AdaptiveSpeculation(max_depth=max_depth, enabled=adaptive)
    begin = clock()
    logical_start = read()
    encoded_start = encoded_bytes_read() if encoded_bytes_read is not None else None
    metrics = dict(mode='greedy_block_verification', rounds=0, proposals=0,
                   accepted_draft_tokens=0, target_forward_calls=0,
                   rollback_tokens=0, draft_seconds=0., verification_seconds=0.,
                   decode_baseline_seconds=None, adaptive=adaptive,
                   round_measurements=[])
    cache = None
    first_time = None
    decode_start_bytes = logical_start
    decode_encoded_start = encoded_start

    def forward(values):
        nonlocal cache
        result = target.forward(np.asarray([values], dtype=np.int32),
                                past_key_values=cache, logits_to_keep=0)
        cache = result.past_key_values
        metrics['target_forward_calls'] += 1
        return choose(result.logits)[0]

    def finish():
        end = clock()
        count = len(tokens) - prefix_length
        encoded_end=encoded_bytes_read() if encoded_bytes_read is not None else None
        encoded_bytes=encoded_end-encoded_start if encoded_start is not None and encoded_end is not None else None
        decode_encoded_bytes=encoded_end-decode_encoded_start if decode_encoded_start is not None and encoded_end is not None else None
        metrics.update(tokens_generated=count, generation_seconds=end-begin,
                       ttft_seconds=first_time-begin if first_time is not None else 0.,
                       decode_tokens_per_second=(count-1)/(end-first_time)
                       if first_time is not None and count > 1 and end > first_time else 0.,
                       logical_bytes=read()-logical_start,
                       logical_bytes_per_accepted_token=(read()-logical_start)/count if count else 0.,
                       decode_logical_bytes_per_accepted_token=(read()-decode_start_bytes)/(count-1)
                       if count > 1 else 0.,
                       encoded_file_bytes=encoded_bytes,
                       encoded_file_bytes_per_accepted_token=encoded_bytes/count
                       if encoded_bytes is not None and count else 0. if encoded_bytes is not None else None,
                       decode_encoded_file_bytes_per_accepted_token=decode_encoded_bytes/(count-1)
                       if decode_encoded_bytes is not None and count>1 else 0. if decode_encoded_bytes is not None else None,
                       acceptance_rate=metrics['accepted_draft_tokens']/metrics['proposals']
                       if metrics['proposals'] else None,
                       disabled_reason=policy.disabled_reason,
                       final_depth=policy.depth,
                       final_cache_position=cache.position if cache is not None else 0)
        return SpeculativeResult(tokens, cache, metrics)

    if not max_new_tokens:
        return finish()
    # Prefill emits the first actual target token, without consulting the draft.
    predictions = forward(tokens)
    pending = int(predictions[-1])
    tokens.append(pending)
    first_time = clock()
    decode_start_bytes = read()
    decode_encoded_start = encoded_bytes_read() if encoded_bytes_read is not None else None
    if pending in eos or max_new_tokens == 1:
        return finish()

    # One real decode pass establishes the same-run autoregressive baseline.
    start = clock()
    pending = int(forward([pending])[-1])
    baseline = max(clock()-start, 1e-12)
    metrics['decode_baseline_seconds'] = baseline
    tokens.append(pending)
    if pending in eos:
        return finish()

    while len(tokens)-prefix_length < max_new_tokens:
        remaining = max_new_tokens-(len(tokens)-prefix_length)
        context_available=getattr(cache,'limit',2**63)-cache.position
        count=min(policy.depth,remaining-1,max(0,context_available-1))
        if not policy.active or count==0:
            start = clock()
            pending = int(forward([pending])[-1])
            baseline = .8*baseline + .2*max(clock()-start, 1e-12)
            tokens.append(pending)
            if pending in eos:
                break
            continue

        round_start = clock()
        read_start = read()
        draft_start = clock()
        # Bound consumption even if a faulty local proposal iterator is infinite.
        candidates = list(islice(iter(propose(list(tokens), count)),count+1))
        draft_seconds = clock()-draft_start
        if len(candidates) > count:
            raise ValueError('Draft returned more candidates than requested')
        if any(not isinstance(t, numbers.Integral) or isinstance(t, bool) or
               not 0 <= int(t) <= np.iinfo(np.int32).max or
               (vocab_size is not None and int(t) >= vocab_size) for t in candidates):
            raise ValueError('Draft returned invalid token ids')
        candidates = [int(t) for t in candidates]
        for j, token in enumerate(candidates):
            if token in eos:
                candidates = candidates[:j+1]
                break
        base = cache.position
        verify_start = clock()
        try:
            # logits[j] predicts candidates[j]; logits[-1] supplies a bonus.
            predictions = forward([pending]+candidates)
            accepted = 0
            for candidate, expected in zip(candidates, predictions):
                if candidate != int(expected):
                    break
                tokens.append(candidate)
                accepted += 1
                if candidate in eos:
                    break
            stop = bool(accepted and tokens[-1] in eos)
            # Accepted EOS is already present in the verified input block. Keep
            # the same lagging-final-token invariant as correction/bonus EOS;
            # otherwise a returned session has one extra committed position.
            committed_position = base+1+accepted-int(stop)
            metrics['rollback_tokens'] += cache.position-committed_position
            cache.truncate(committed_position)
            if not stop:
                pending = int(predictions[accepted])
                tokens.append(pending)
                stop = pending in eos
        except BaseException:
            # forward itself restores its entry position; preserve that invariant
            # even if a backend fails after returning a partially populated cache.
            cache.truncate(base)
            raise
        verify_seconds = clock()-verify_start
        elapsed = clock()-round_start
        # EOS accepted above has no bonus. Correction/bonus EOS still counts.
        if accepted and candidates[accepted-1] in eos:
            emitted = accepted
        else:
            emitted = accepted+1
        metrics['rounds'] += 1
        metrics['proposals'] += len(candidates)
        metrics['accepted_draft_tokens'] += accepted
        metrics['draft_seconds'] += draft_seconds
        metrics['verification_seconds'] += verify_seconds
        metrics['round_measurements'].append(dict(
            depth=count, proposed=len(candidates), accepted=accepted, emitted=emitted,
            draft_seconds=draft_seconds, verify_seconds=verify_seconds,
            total_seconds=elapsed, logical_bytes=read()-read_start,
            logical_bytes_per_accepted_token=(read()-read_start)/emitted))
        policy.observe(seconds=elapsed, emitted=emitted, proposed=len(candidates),
                       accepted=accepted, baseline_seconds=baseline)
        if stop:
            break
    return finish()
