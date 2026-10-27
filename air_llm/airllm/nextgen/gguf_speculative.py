"""Opt-in greedy block verification for the GGUF Qwen hybrid runtime.

Recurrent state cannot be truncated by token offset. A rejected verification
restores a budgeted snapshot of all KV, convolution and DeltaNet banks, then
replays the accepted input prefix. Replay is a real target pass: its elapsed
time, weight reads and expert work remain in the adaptive cost observations.
No intermediate recurrent histories, model downloads or new quantization are
introduced. Block/serial floating-point parity still requires checkpoint tests.
"""
from time import perf_counter
from types import SimpleNamespace
import traceback

import numpy as np

from .generation import eos_ids, generation_options, token_batch
from .prompt_draft import PromptSpanDraft
from .speculative import LocalModelDraft, greedy_speculative_generate
from .storage import encoded_read_bytes, physical_read_bytes


_SNAPSHOT_BUDGET_ERRORS = frozenset({
    'Qwen rollback snapshot exceeds kv_bytes',
    'Qwen context plus retained rollback states exceeds kv_bytes',
})


class SnapshotVerifier:
    """Adapt Qwen's immutable snapshot banks to the existing verifier contract.

    Only one external snapshot is retained, inside the session's existing
    budget. The runtime's own transactional snapshot remains authoritative.
    Fully accepted blocks release the snapshot without another target pass.
    The returned cache keeps the verifier's lagging-final-token invariant.
    """
    def __init__(self, model, *, clock=perf_counter):
        self.model = model
        self.clock = clock
        self.session = None
        self.snapshot = None
        self.inputs = ()
        self.stats = dict(actual_target_forward_calls=0,
                          actual_target_forward_positions=0,
                          recurrent_replay_calls=0,
                          recurrent_replay_positions=0,
                          recurrent_replay_seconds=0.,
                          recurrent_snapshot_restores=0)

    @property
    def position(self):
        return self.session.position if self.session is not None else 0

    @property
    def limit(self):
        return self.session.limit

    @property
    def bytes(self):
        return self.session.bytes

    @property
    def peak_bytes(self):
        return self.session.peak_bytes

    def release(self):
        if self.snapshot is not None:
            self.session.release(self.snapshot)
            self.snapshot = None
        self.inputs = ()

    def _forward(self, values, logits_to_keep):
        # Count attempted work as well as successful work; failures must never
        # make expensive target reads disappear from the accounting.
        self.stats['actual_target_forward_calls'] += 1
        self.stats['actual_target_forward_positions'] += len(values)
        return self.model.forward(np.asarray([values], dtype=np.int32),
                                  past_key_values=self.session,
                                  logits_to_keep=logits_to_keep)

    def forward(self, input_ids, past_key_values=None, logits_to_keep=0):
        if past_key_values is not None and past_key_values is not self:
            raise ValueError('Verifier cache belongs to another target')
        ids = token_batch(input_ids, self.model.config['vocab_size'])
        self.release()
        if self.session is None:
            self.session = self.model.new_session()
        self.snapshot = self.session.snapshot()
        self.inputs = tuple(map(int, ids[0]))
        try:
            result = self._forward(self.inputs, logits_to_keep)
            if result.past_key_values is not self.session:
                raise ValueError('Qwen forward replaced its session')
            return SimpleNamespace(logits=result.logits, past_key_values=self)
        except BaseException:
            # The engine normally already restores its internal snapshot.
            # This outer snapshot also protects errors after forward returns.
            self.session.restore(self.snapshot)
            self.snapshot = None
            self.inputs = ()
            raise

    def truncate(self, position):
        if type(position) is not int:
            raise ValueError('Committed position must be an integer')
        if position == self.position:
            self.release()
            return
        if (self.snapshot is None or
                not self.snapshot.position <= position <= self.position):
            raise ValueError('Committed position is outside the verified block')
        start = self.snapshot.position
        committed_inputs = self.inputs[:position-start]
        self.session.restore(self.snapshot)
        self.snapshot = None
        self.inputs = ()
        self.stats['recurrent_snapshot_restores'] += 1
        if committed_inputs:
            begin = self.clock()
            self.stats['recurrent_replay_calls'] += 1
            self.stats['recurrent_replay_positions'] += len(committed_inputs)
            try:
                self._forward(committed_inputs, 1)
            finally:
                self.stats['recurrent_replay_seconds'] += self.clock()-begin


def _draft_proposer(model, draft_model, eos):
    if draft_model is None:
        draft_model = PromptSpanDraft()
    if draft_model is model:
        raise ValueError('Draft and target must be distinct local models')
    if hasattr(draft_model, 'generate'):
        config = getattr(draft_model, 'config', {})
        vocab = (config.get('vocab_size') if isinstance(config, dict)
                 else getattr(config, 'vocab_size', None))
        if vocab is not None and vocab != model.config['vocab_size']:
            raise ValueError('Draft and target vocabularies differ')
        a = getattr(model, 'tokenizer', None)
        b = getattr(draft_model, 'tokenizer', None)
        if a is not None and b is not None and a.get_vocab() != b.get_vocab():
            raise ValueError('Draft and target token mappings differ')
        propose = LocalModelDraft(draft_model, eos)
    elif hasattr(draft_model, 'propose'):
        propose = draft_model.propose
    elif callable(draft_model):
        propose = draft_model
    else:
        raise TypeError('Supply a local draft model or proposal callable')
    checkpoint = getattr(draft_model, 'checkpoint', None)
    logical = getattr(draft_model, 'logical_bytes_read', None)
    encoded = getattr(draft_model, 'encoded_bytes_read', None)
    if checkpoint is not None:
        logical = lambda: checkpoint.stats['logical_bytes']
        encoded = lambda: encoded_read_bytes(checkpoint)
    return (propose, logical if callable(logical) else None,
            encoded if callable(encoded) else None)


def generate_gguf_speculative(model, input_ids, draft_model=None, *,
                              max_new_tokens=32, eos_token_id=None,
                              max_depth=4, adaptive=True, candidate_depths=None,
                              temperature=0., retry_empty_drafts=False):
    """Explicit experimental entry point; ordinary generation stays unchanged.

    The default draft retrieves from the known token prefix without disk I/O.
    Callables must use the target's token IDs. A supplied model must already be
    local and loaded; equal vocabulary sizes alone do not prove token mapping
    compatibility when tokenizers are unavailable. Depth counts *proposals*;
    target verification processes the pending token plus up to that depth.
    A recognized recurrent snapshot budget refusal restarts ordinary generation
    from the original input. No output has been returned at that point. Failed
    speculative work and the repeated prefill remain charged to the request.
    """
    max_new_tokens, temperature = generation_options(max_new_tokens, temperature)
    if temperature != 0:
        raise ValueError('GGUF speculative decoding supports greedy temperature=0 only')
    host = token_batch(input_ids, model.config['vocab_size'])
    eos = eos_ids(model.config.get('eos_token_id') if eos_token_id is None
                  else eos_token_id, model.config['vocab_size'])
    propose, draft_read, draft_encoded = _draft_proposer(model, draft_model, eos)
    with model.lock:
        if model.closed:
            raise RuntimeError('Model is closed')
        request_begin = perf_counter()
        logical_start = model.checkpoint.stats['logical_bytes']
        encoded_start = encoded_read_bytes(model.checkpoint)
        draft_logical_start = draft_read() if draft_read is not None else None
        draft_encoded_start = draft_encoded() if draft_encoded is not None else None
        verifier = SnapshotVerifier(model)
        physical = physical_read_bytes()
        def check_finite(logits):
            with model.mx.stream(model.stream):
                if not bool(model.mx.all(model.mx.isfinite(logits)).item()):
                    raise ValueError('GGUF speculative target returned non-finite logits')
        def argmax(logits):
            check_finite(logits)
            with model.mx.stream(model.stream):
                return model.mx.argmax(logits, axis=-1).tolist()
        failure = None
        verification_begin = perf_counter()
        try:
            result = greedy_speculative_generate(
                verifier, propose, host, max_new_tokens,
                eos_token_ids=eos, max_depth=max_depth, adaptive=adaptive,
                argmax_tokens=argmax, vocab_size=model.config['vocab_size'],
                bytes_read=lambda: model.checkpoint.stats['logical_bytes'],
                encoded_bytes_read=lambda: encoded_read_bytes(model.checkpoint),
                draft_bytes_read=draft_read, draft_encoded_bytes_read=draft_encoded,
                candidate_depths=candidate_depths,
                retry_empty_drafts=retry_empty_drafts,
                max_verify_positions=max(1, model.options.workspace_bytes //
                                         (8*model.config['vocab_size'])),
                forward_metrics=lambda: dict(model.reader.stats))
        except BaseException as error:
            failure = error
        finally:
            verifier.release()
        def transport_totals():
            logical = model.checkpoint.stats['logical_bytes']-logical_start
            encoded = encoded_read_bytes(model.checkpoint)-encoded_start
            draft_logical = (draft_read()-draft_logical_start
                             if draft_logical_start is not None else None)
            draft_file = (draft_encoded()-draft_encoded_start
                          if draft_encoded_start is not None else None)
            return dict(logical_bytes=logical, encoded_file_bytes=encoded,
                        draft_logical_bytes=draft_logical,
                        draft_encoded_file_bytes=draft_file,
                        total_logical_bytes=logical+draft_logical
                            if draft_logical is not None else None,
                        total_encoded_file_bytes=encoded+draft_file
                            if draft_file is not None else None)
        def failure_metrics(error, **extra):
            model.stats['speculative'] = dict(
                status='failed', failure_type=type(error).__name__,
                failure_message=str(error),
                generation_seconds=perf_counter()-request_begin,
                final_cache_position=verifier.position,
                retained_snapshot_bytes=verifier.session.retained
                    if verifier.session is not None else 0,
                **verifier.stats, **transport_totals(), **extra)
        if failure is not None:
            if not (isinstance(failure, MemoryError) and
                    str(failure) in _SNAPSHOT_BUDGET_ERRORS):
                failure_metrics(failure)
                raise failure
            fallback_reason = str(failure)
            abandoned_position = verifier.position
            abandoned_peak = verifier.peak_bytes if verifier.session else 0
            # The failed exception's traceback retains the generator frame,
            # its output arrays and recurrent cache. Releasing its snapshot
            # alone is insufficient: drop all old-bank aliases before starting
            # a second cache. Only this recognized, handled budget exception is
            # cleared; unrelated failures retain their original tracebacks.
            try:
                model.reader.drain()
            except BaseException as drain_error:
                failure_metrics(drain_error, fallback_attempted=False,
                                fallback_reason=fallback_reason)
                raise
            traceback.clear_frames(failure.__traceback__)
            failure.__traceback__ = None
            failure.__cause__ = None
            failure.__context__ = None
            failure = None
            verifier.session = None
            abandoned_seconds = perf_counter()-request_begin
            abandoned_transport = transport_totals()
            # Install a reversible finite-logit guard while holding the same
            # model lock. Ordinary generation otherwise has no such check.
            # Existing harness wrappers remain in the call chain and the
            # exact previous instance/class method binding is restored.
            ordinary_forward = model.forward
            had_instance_forward = 'forward' in vars(model)
            def finite_forward(*args, **kwargs):
                value = ordinary_forward(*args, **kwargs)
                check_finite(value.logits)
                return value
            model.forward = finite_forward
            try:
                answer = model.generate(host, max_new_tokens=max_new_tokens,
                                        eos_token_id=list(eos), temperature=0.)
            except BaseException as fallback_error:
                failure_metrics(fallback_error, fallback_attempted=True,
                    fallback_reason=fallback_reason,
                    abandoned_speculative_cache_position=abandoned_position,
                    abandoned_speculative_peak_bytes=abandoned_peak,
                    abandoned_speculative_seconds=abandoned_seconds,
                    abandoned_speculative_transport=abandoned_transport)
                raise
            finally:
                if had_instance_forward:
                    model.forward = ordinary_forward
                else:
                    del model.forward
            ordinary = {key: value for key, value in model.stats.items()
                        if key != 'speculative'}
            measured = dict(status='ordinary_after_snapshot_budget_refusal',
                fallback_attempted=True, fallback_reason=fallback_reason,
                fallback_finite_logits_checked=True,
                abandoned_speculative_cache_position=abandoned_position,
                abandoned_speculative_peak_bytes=abandoned_peak,
                abandoned_speculative_seconds=abandoned_seconds,
                abandoned_speculative_transport=abandoned_transport,
                ordinary_runtime=ordinary,
                actual_target_forward_calls_scope='abandoned_speculative_attempt_only',
                ordinary_target_forward_calls=ordinary['tokens_generated'],
                generation_seconds=perf_counter()-request_begin,
                ttft_seconds=abandoned_seconds+ordinary['ttft_seconds'],
                tokens_generated=ordinary['tokens_generated'],
                decode_tokens_per_second=ordinary['decode_tokens_per_second'],
                final_cache_position=len(host[0])+ordinary['tokens_generated']-1,
                retained_snapshot_bytes=0,
                full_checkpoint_token_parity_validated=False,
                **verifier.stats, **transport_totals())
            count = measured['tokens_generated']
            measured['logical_bytes_per_accepted_token'] = measured['logical_bytes']/count if count else 0.
            measured['encoded_file_bytes_per_accepted_token'] = measured['encoded_file_bytes']/count if count else 0.
            model.stats.update(generation_seconds=measured['generation_seconds'],
                ttft_seconds=measured['ttft_seconds'],
                logical_bytes_per_generated_token=measured['logical_bytes_per_accepted_token'],
                encoded_file_bytes_per_generated_token=measured['encoded_file_bytes_per_accepted_token'],
                context_payload_peak_bytes=max(ordinary.get('context_payload_peak_bytes', 0),
                                              abandoned_peak),
                speculative=measured)
            physical_end = physical_read_bytes()
            model.stats['physical_read_bytes'] = (physical_end-physical
                if physical is not None and physical_end is not None else None)
            return answer
        measured = result.metrics
        measured['status'] = 'completed'
        measured['ttft_seconds'] += verification_begin-request_begin if measured['tokens_generated'] else 0.
        measured['generation_seconds'] = perf_counter()-request_begin
        measured.update(verifier.stats)
        measured.update(recurrent_rollback='snapshot_restore_and_accepted_prefix_replay',
                        full_checkpoint_token_parity_validated=False,
                        block_serial_rounding_note='Checkpoint token/logit parity must be measured',
                        retained_snapshot_bytes=verifier.session.retained
                            if verifier.session is not None else 0)
        model.stats.update(tokens_generated=measured['tokens_generated'],
                           generation_seconds=measured['generation_seconds'],
                           ttft_seconds=measured['ttft_seconds'],
                           decode_tokens_per_second=measured['decode_tokens_per_second'],
                           logical_bytes_per_generated_token=measured['logical_bytes_per_accepted_token'],
                           encoded_file_bytes_per_generated_token=measured['encoded_file_bytes_per_accepted_token'],
                           context_payload_peak_bytes=verifier.peak_bytes if verifier.session else 0,
                           speculative=measured)
        physical_end = physical_read_bytes()
        model.stats['physical_read_bytes'] = (physical_end-physical
            if physical is not None and physical_end is not None else None)
        with model.mx.stream(model.stream):
            return model.mx.array([result.tokens], dtype=model.mx.int32)
