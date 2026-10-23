"""Exact greedy draft-and-verify decoding with append-only KV rollback.

Every emitted token is the target's argmax under its accepted prefix. A draft
only proposes inputs for one causal target forward; it never supplies an
unverified answer. Sampling and tree verification are intentionally unsupported.
The draft is an explicitly supplied local model/callable; no download or service
lookup occurs here. Block evaluation can have ordinary floating-point rounding
differences from one-token evaluation, as with any batched inference.
"""
from dataclasses import dataclass
from collections import defaultdict, deque
from itertools import islice
from time import perf_counter
from statistics import median_low
import math
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
    """Bounded online depth trials using measured time, then transport cost.

    Complete draft+verification time is the primary objective. Transport bytes
    break ties only between depths within 3% of the best measured time/token;
    a byte reduction can never excuse a measured slowdown beyond the guard.
    Samples are from this generation, not a presumed future expert route.
    Single-token calibration is refreshed while speculation remains active.
    """
    max_depth: int = 4
    enabled: bool = True
    warmup_rounds: int = 2
    slow_rounds_to_disable: int = 2
    slowdown_tolerance: float = 1.05
    calibration_interval: int = 4
    history_window: int = 8
    candidate_depths: tuple[int, ...] | None = None

    def __post_init__(self):
        for name in ('max_depth', 'slow_rounds_to_disable', 'calibration_interval', 'history_window'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.warmup_rounds) is not int or self.warmup_rounds < 0:
            raise ValueError('warmup_rounds must be a nonnegative integer')
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be boolean')
        if (type(self.slowdown_tolerance) not in (int, float)
                or not math.isfinite(self.slowdown_tolerance) or self.slowdown_tolerance < 1):
            raise ValueError('slowdown_tolerance must be finite and at least one')
        self.depth = min(2, self.max_depth) if self.enabled else self.max_depth
        if self.candidate_depths is not None:
            if (not self.candidate_depths or any(type(d) is not int or not 1 <= d <= self.max_depth
                    for d in self.candidate_depths) or tuple(sorted(set(self.candidate_depths))) != self.candidate_depths):
                raise ValueError('candidate_depths must be sorted unique positive depths within max_depth')
            if self.enabled:self.depth = self.candidate_depths[0]
        self.active = True
        self.rounds = 0
        self.slow_streak = 0
        self.total_slow_streak = 0
        self.zero_accept_streak = 0
        self.empty_streak = 0
        self.disabled_reason = None
        self.costs = defaultdict(lambda: deque(maxlen=self.history_window))
        self.baselines = deque(maxlen=self.history_window)
        self.rounds_since_calibration = 0
        self.depth_slow_streak = defaultdict(int)
        self.blocked_depths = set()

    def record_baseline(self, seconds, logical_bytes=None, encoded_bytes=None):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Baseline seconds must be finite and nonnegative')
        for value in (logical_bytes, encoded_bytes):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('Transport observations must be nonnegative integers or unknown')
        self.baselines.append((max(seconds, 1e-12), logical_bytes, encoded_bytes))
        self.rounds_since_calibration = 0

    @property
    def baseline_seconds(self):
        # The lower median is conservative with two samples: a cold first pass
        # cannot inflate the reference and authorize persistently slow drafts.
        return median_low(s[0] for s in self.baselines) if self.baselines else None

    @property
    def needs_calibration(self):
        return (self.enabled and self.active
                and self.rounds_since_calibration >= self.calibration_interval)

    def cost_summary(self):
        result = {}
        for depth, samples in self.costs.items():
            emitted = sum(s['emitted'] for s in samples)
            def rate(name):
                return (sum(s[name] for s in samples)/emitted
                        if all(s[name] is not None for s in samples) else None)
            result[str(depth)] = dict(samples=len(samples), emitted=emitted,
                seconds_per_token=rate('seconds'),
                logical_bytes_per_token=rate('logical_bytes'),
                encoded_file_bytes_per_token=rate('encoded_bytes'),
                blocked=depth in self.blocked_depths)
        return result

    def _select_measured_depth(self, trial_depth, fully_accepted, baseline):
        summaries = self.cost_summary()
        viable = {int(d): s for d, s in summaries.items()
                  if int(d) not in self.blocked_depths and
                  (self.candidate_depths is None or int(d) in self.candidate_depths)}
        # Two observations at a fully accepted depth permit one neighboring
        # trial. This is bounded exploration, not an assumed acceleration.
        summary = summaries[str(trial_depth)]
        next_depth = (next((d for d in self.candidate_depths if d > trial_depth), self.max_depth+1)
                      if self.candidate_depths else trial_depth+1)
        if (fully_accepted and summary['samples'] >= 2 and next_depth <= self.max_depth
                and next_depth not in self.blocked_depths and str(next_depth) not in summaries
                and (baseline is None or summary['seconds_per_token'] <= baseline*self.slowdown_tolerance)):
            self.depth = next_depth
            return
        if not viable:
            self.depth = self._reduced_depth(trial_depth)
            return
        fastest = min(s['seconds_per_token'] for s in viable.values())
        near = [(d, s) for d, s in viable.items() if s['seconds_per_token'] <= fastest*1.03]
        # Compare one byte domain across all candidates. An encoded file byte
        # count cannot be compared with another depth's decoded/logical bytes.
        domain = ('encoded_file_bytes_per_token'
                  if all(s['encoded_file_bytes_per_token'] is not None for _,s in near)
                  else 'logical_bytes_per_token'
                  if all(s['logical_bytes_per_token'] is not None for _,s in near) else None)
        def key(item):
            d, s = item
            return (s[domain] if domain else 0., s['seconds_per_token'], d)
        self.depth = min(near, key=key)[0]

    def _reduced_depth(self, trial_depth):
        limit=max(1,min(self.max_depth,trial_depth//2))
        if self.candidate_depths:
            return max((d for d in self.candidate_depths if d <= limit),
                       default=self.candidate_depths[0])
        return limit

    def observe(self, *, seconds, emitted, proposed, accepted, baseline_seconds,
                depth=None, logical_bytes=None, encoded_bytes=None):
        if (not math.isfinite(seconds) or seconds < 0 or type(emitted) is not int or emitted < 1
                or type(proposed) is not int or proposed < 0 or type(accepted) is not int
                or not 0 <= accepted <= proposed):
            raise ValueError('Invalid speculative cost observation')
        for value in (logical_bytes, encoded_bytes):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('Transport observations must be nonnegative integers or unknown')
        if baseline_seconds is not None and (not math.isfinite(baseline_seconds) or baseline_seconds <= 0):
            raise ValueError('Baseline seconds must be positive or unknown')
        trial_depth = self.depth if depth is None else depth
        if type(trial_depth) is not int or not 1 <= trial_depth <= self.max_depth:
            raise ValueError('Observed depth is outside the configured range')
        self.costs[trial_depth].append(dict(seconds=seconds, emitted=emitted,
            logical_bytes=logical_bytes, encoded_bytes=encoded_bytes))
        self.rounds += 1
        self.rounds_since_calibration += 1
        if not self.enabled:
            return
        self.zero_accept_streak = self.zero_accept_streak+1 if proposed and accepted==0 else 0
        self.empty_streak = self.empty_streak+1 if not proposed else 0
        self._select_measured_depth(trial_depth, bool(proposed and accepted == proposed), baseline_seconds)
        if proposed and accepted * 2 < proposed:
            self.depth = min(self.depth, self._reduced_depth(trial_depth))
        # Repeatedly accepting no draft tokens cannot amortize any target pass.
        # This observed progress guard remains valid even when a noisy initial
        # decode timing badly overestimates the wall-time fallback baseline.
        if self.zero_accept_streak >= self.warmup_rounds+self.slow_rounds_to_disable:
            self.active = False
            self.disabled_reason = 'no_accepted_draft_tokens'
            return
        if self.empty_streak >= self.warmup_rounds+self.slow_rounds_to_disable:
            self.active = False
            self.disabled_reason = 'empty_draft_proposals'
            return
        if self.rounds <= self.warmup_rounds or baseline_seconds is None:
            return
        slow = seconds / max(1, emitted) > baseline_seconds * self.slowdown_tolerance
        self.depth_slow_streak[trial_depth] = self.depth_slow_streak[trial_depth]+1 if slow else 0
        if self.depth_slow_streak[trial_depth] >= self.slow_rounds_to_disable:
            self.blocked_depths.add(trial_depth)
            self.depth = self._reduced_depth(trial_depth)
        self.slow_streak = self.slow_streak + 1 if slow else 0
        if self.slow_streak >= self.slow_rounds_to_disable:
            self.active = False
            self.disabled_reason = 'measured_draft_plus_verify_cost_exceeds_decode_baseline'

    def observe_total_decode(self, *, seconds, emitted, baseline_seconds):
        """Guard observed decode time including calibration and bookkeeping."""
        if (not math.isfinite(seconds) or seconds < 0 or type(emitted) is not int
                or emitted < 1 or not math.isfinite(baseline_seconds) or baseline_seconds <= 0):
            raise ValueError('Invalid total decode cost observation')
        if not self.enabled or not self.active or self.rounds <= self.warmup_rounds:return
        slow=seconds/emitted > baseline_seconds*self.slowdown_tolerance
        self.total_slow_streak=self.total_slow_streak+1 if slow else 0
        if self.total_slow_streak >= self.slow_rounds_to_disable:
            self.active=False
            self.disabled_reason='measured_total_decode_cost_exceeds_decode_baseline'


def _host_argmax(logits):
    return np.argmax(np.asarray(logits), axis=-1).tolist()


def greedy_speculative_generate(target, propose, input_ids, max_new_tokens,
                                *, eos_token_ids=(), max_depth=4, adaptive=True,
                                argmax_tokens=None, bytes_read=None,
                                clock=perf_counter,vocab_size=None,encoded_bytes_read=None,
                                draft_bytes_read=None,draft_encoded_bytes_read=None,
                                candidate_depths=None, forward_metrics=None,
                                max_verify_positions=None):
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
    policy = AdaptiveSpeculation(max_depth=max_depth, enabled=adaptive,
                                candidate_depths=candidate_depths)
    if max_verify_positions is not None and (type(max_verify_positions) is not int or max_verify_positions < 1):
        raise ValueError('max_verify_positions must be a positive integer or None')
    begin = clock()
    logical_start = read()
    encoded_start = encoded_bytes_read() if encoded_bytes_read is not None else None
    draft_logical_start = draft_bytes_read() if draft_bytes_read is not None else None
    draft_encoded_start = draft_encoded_bytes_read() if draft_encoded_bytes_read is not None else None
    metrics = dict(mode='greedy_block_verification', rounds=0, proposals=0,
                   accepted_draft_tokens=0, target_forward_calls=0,
                   target_forward_positions=0,draft_calls=0,rejected_proposal_tokens=0,
                   rollback_tokens=0, draft_seconds=0., verification_seconds=0.,
                   decode_baseline_seconds=None, adaptive=adaptive,
                   calibration_forward_calls=0, ordinary_decode_forward_calls=0,
                   baseline_measurements=[],
                   round_measurements=[])
    cache = None
    first_time = None
    decode_start_bytes = logical_start
    decode_encoded_start = encoded_start

    def forward(values, keep=0):
        nonlocal cache
        result = target.forward(np.asarray([values], dtype=np.int32),
                                past_key_values=cache, logits_to_keep=keep)
        cache = result.past_key_values
        metrics['target_forward_calls'] += 1
        metrics['target_forward_positions'] += len(values)
        return choose(result.logits)[0]

    def finish():
        end = clock()
        count = len(tokens) - prefix_length
        encoded_end=encoded_bytes_read() if encoded_bytes_read is not None else None
        encoded_bytes=encoded_end-encoded_start if encoded_start is not None and encoded_end is not None else None
        decode_encoded_bytes=encoded_end-decode_encoded_start if decode_encoded_start is not None and encoded_end is not None else None
        draft_logical_bytes=draft_bytes_read()-draft_logical_start if draft_logical_start is not None else None
        draft_encoded_bytes=draft_encoded_bytes_read()-draft_encoded_start if draft_encoded_start is not None else None
        metrics.update(tokens_generated=count, generation_seconds=end-begin,
                       ttft_seconds=first_time-begin if first_time is not None else 0.,
                       decode_tokens_per_second=(count-1)/(end-first_time)
                       if first_time is not None and count > 1 and end > first_time else 0.,
                       decode_seconds_per_token=(end-first_time)/(count-1)
                       if first_time is not None and count>1 else None,
                       logical_bytes=read()-logical_start,
                       target_logical_bytes_observed=bytes_read is not None,
                       logical_bytes_per_accepted_token=(read()-logical_start)/count if count else 0.,
                       decode_logical_bytes_per_accepted_token=(read()-decode_start_bytes)/(count-1)
                       if count > 1 else 0.,
                       encoded_file_bytes=encoded_bytes,
                       draft_logical_bytes=draft_logical_bytes,
                       draft_encoded_file_bytes=draft_encoded_bytes,
                       total_logical_bytes=(read()-logical_start+draft_logical_bytes)
                       if draft_logical_bytes is not None and bytes_read is not None else None,
                       total_encoded_file_bytes=encoded_bytes+draft_encoded_bytes
                       if encoded_bytes is not None and draft_encoded_bytes is not None else None,
                       encoded_file_bytes_per_accepted_token=encoded_bytes/count
                       if encoded_bytes is not None and count else 0. if encoded_bytes is not None else None,
                       decode_encoded_file_bytes_per_accepted_token=decode_encoded_bytes/(count-1)
                       if decode_encoded_bytes is not None and count>1 else 0. if decode_encoded_bytes is not None else None,
                       acceptance_rate=metrics['accepted_draft_tokens']/metrics['proposals']
                       if metrics['proposals'] else None,
                       disabled_reason=policy.disabled_reason,
                       depth_costs=policy.cost_summary(),
                       cost_observation_scope=('target_and_draft_transport_unknown' if bytes_read is None
                                               else 'target_and_draft' if draft_bytes_read is not None
                                               else 'target_only_draft_transport_unknown'),
                       final_depth=policy.depth,
                       final_cache_position=cache.position if cache is not None else 0)
        return SpeculativeResult(tokens, cache, metrics)

    if not max_new_tokens:
        return finish()
    # Prefill emits the first actual target token, without consulting the draft.
    predictions = forward(tokens, keep=1)
    pending = int(predictions[-1])
    tokens.append(pending)
    first_time = clock()
    decode_start_bytes = read()
    decode_encoded_start = encoded_bytes_read() if encoded_bytes_read is not None else None
    if pending in eos or max_new_tokens == 1:
        return finish()

    def ordinary_decode(*, calibration=False):
        nonlocal pending
        start = clock()
        before_bytes = read()
        before_encoded = encoded_bytes_read() if encoded_bytes_read is not None else None
        pending = int(forward([pending])[-1])
        seconds = max(clock()-start, 1e-12)
        logical = read()-before_bytes
        encoded = encoded_bytes_read()-before_encoded if before_encoded is not None else None
        policy.record_baseline(seconds,logical,encoded)
        metrics['baseline_measurements'].append(dict(seconds=seconds,logical_bytes=logical,
                                                    encoded_file_bytes=encoded,
                                                    calibration=calibration))
        metrics['ordinary_decode_forward_calls'] += 1
        metrics['calibration_forward_calls'] += int(calibration)
        metrics['decode_baseline_seconds'] = policy.baseline_seconds
        return policy.baseline_seconds

    # One real decode pass establishes the same-run autoregressive baseline.
    baseline = ordinary_decode(calibration=True)
    tokens.append(pending)
    if pending in eos:
        return finish()

    while len(tokens)-prefix_length < max_new_tokens:
        remaining = max_new_tokens-(len(tokens)-prefix_length)
        context_available=getattr(cache,'limit',2**63)-cache.position
        count=min(policy.depth,remaining-1,max(0,context_available-1))
        if max_verify_positions is not None:count=min(count,max_verify_positions-1)
        calibrating=bool(policy.needs_calibration and count)
        if not policy.active or count==0 or calibrating:
            baseline = ordinary_decode(calibration=calibrating)
            tokens.append(pending)
            if pending in eos:
                break
            continue

        round_start = clock()
        read_start = read()
        encoded_round_start = encoded_bytes_read() if encoded_bytes_read is not None else None
        draft_round_start = draft_bytes_read() if draft_bytes_read is not None else None
        draft_encoded_round_start = draft_encoded_bytes_read() if draft_encoded_bytes_read is not None else None
        draft_start = clock()
        # Bound consumption even if a faulty local proposal iterator is infinite.
        candidates = list(islice(iter(propose(list(tokens), count)),count+1))
        metrics['draft_calls'] += 1
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
        forward_before = forward_metrics() if forward_metrics is not None else None
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
        forward_after = forward_metrics() if forward_metrics is not None else None
        forward_delta = ({k: v-forward_before.get(k, 0) for k,v in forward_after.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool)}
                         if forward_after is not None else None)
        elapsed = clock()-round_start
        # EOS accepted above has no bonus. Correction/bonus EOS still counts.
        if accepted and candidates[accepted-1] in eos:
            emitted = accepted
        else:
            emitted = accepted+1
        metrics['rounds'] += 1
        metrics['proposals'] += len(candidates)
        metrics['rejected_proposal_tokens'] += len(candidates)-accepted
        metrics['accepted_draft_tokens'] += accepted
        metrics['draft_seconds'] += draft_seconds
        metrics['verification_seconds'] += verify_seconds
        logical = read()-read_start
        encoded = encoded_bytes_read()-encoded_round_start if encoded_round_start is not None else None
        draft_logical = draft_bytes_read()-draft_round_start if draft_round_start is not None else None
        draft_encoded = draft_encoded_bytes_read()-draft_encoded_round_start if draft_encoded_round_start is not None else None
        total_logical = logical+draft_logical if draft_logical is not None and bytes_read is not None else None
        total_encoded = encoded+draft_encoded if encoded is not None and draft_encoded is not None else None
        metrics['round_measurements'].append(dict(
            depth=count, proposed=len(candidates), accepted=accepted, emitted=emitted,
            draft_seconds=draft_seconds, verify_seconds=verify_seconds,
            total_seconds=elapsed, logical_bytes=logical,
            logical_bytes_per_accepted_token=logical/emitted,
            encoded_file_bytes=encoded, draft_logical_bytes=draft_logical,
            draft_encoded_file_bytes=draft_encoded, total_logical_bytes=total_logical,
            total_encoded_file_bytes=total_encoded,
            rejected_proposal_tokens=len(candidates)-accepted,
            verification_input_tokens=len(candidates)+1,
            expert_batch_delta=forward_delta,
            kv_bytes=getattr(cache,'bytes',None), kv_peak_bytes=getattr(cache,'peak_bytes',None),
            baseline_seconds=baseline, next_depth=None))
        policy.observe(seconds=elapsed, emitted=emitted, proposed=len(candidates),
                       accepted=accepted, baseline_seconds=baseline,depth=count,
                       logical_bytes=(total_logical if total_logical is not None else logical)
                       if bytes_read is not None else None,
                       encoded_bytes=total_encoded if total_encoded is not None else encoded)
        # Service-cost depth choices alone can miss their own bookkeeping or
        # calibration overhead. Judge the observed full decode path as well.
        decode_elapsed=clock()-first_time
        decode_emitted=len(tokens)-prefix_length-1
        policy.observe_total_decode(seconds=decode_elapsed,emitted=decode_emitted,
                                    baseline_seconds=baseline)
        metrics['last_policy_observed_decode_seconds_per_token']=decode_elapsed/decode_emitted
        metrics['round_measurements'][-1]['next_depth']=policy.depth
        if stop:
            break
    return finish()
