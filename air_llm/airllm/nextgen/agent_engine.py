"""Bounded, cooperative local-agent inference over one existing Next model.

Only one request owns a resident architecture session. Switching discards that
session and reconstructs it from a bounded token transcript using the original
prefill followed by singleton decode, verifying every recorded greedy token.
This deliberately trades replay time for a single KV/recurrent memory budget.
No model weights are duplicated, sampled decoding or external tools executed.
"""
from dataclasses import dataclass, field
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import tempfile
from threading import get_ident
from time import perf_counter, time
from uuid import uuid4


_JOB_ID = re.compile(r'^[a-f0-9]{32}$')
_TERMINAL = frozenset(('completed', 'cancelled', 'failed'))
_STATES = frozenset(('queued', 'running', 'paused', 'completed', 'cancelled', 'failed'))


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} is outside its integer bounds')
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _tokens(value, vocab_size, name, nonempty=False):
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise ValueError(f'{name} must be a token list')
    if any(type(v) is not int or not 0 <= v < min(vocab_size, 2**31) for v in value):
        raise ValueError(f'{name} contains an invalid token')
    return list(value)


@dataclass
class AgentJob:
    id: str
    agent_id: str
    input_ids: list
    max_tokens: int
    priority: int
    eos_token_ids: list
    sequence: int
    state: str = 'queued'
    output_ids: list = field(default_factory=list)
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)
    metrics: dict = field(default_factory=lambda: {
        'inference_seconds': 0., 'replay_seconds': 0., 'replayed_tokens': 0,
        'logical_bytes': 0, 'encoded_file_bytes': 0, 'quanta': 0,
        'first_token_latency_seconds': None, 'completion_latency_seconds': None,
        'resident_context_peak_bytes': 0, 'failed_quanta': 0, 'discarded_tokens': 0})
    error: str | None = None


class AgentEngine:
    """Owner-thread priority queue with durable, replayable greedy transcripts.

    Higher integer priority runs first. Equal priorities rotate at quantum
    boundaries. A quantum cannot interrupt an in-flight forward/pread/kernel;
    cancellation latency includes that work. Transcript reservations account
    conservatively for Python integer/list storage (96 bytes/token + 4096/job).
    Native allocator/weight/one-context budgets remain the model's NextConfig.
    Journals contain tokens, not KV arrays, and are fsynced once per quantum.
    """
    VERSION = 1

    def __init__(self, model, *, state_dir=None, max_requests=64,
                 max_transcript_bytes=32*1024**2, max_journal_bytes=32*1024**2,
                 max_journal_write_bytes=256*1024**2, quantum_tokens=8):
        self.owner = get_ident()
        self.model = model
        self.max_requests = _integer(max_requests, 'max_requests', 1)
        self.max_transcript_bytes = _integer(max_transcript_bytes, 'max_transcript_bytes', 1)
        self.max_journal_bytes = _integer(max_journal_bytes, 'max_journal_bytes', 1)
        self.max_journal_write_bytes = _integer(max_journal_write_bytes, 'max_journal_write_bytes', 1)
        self.quantum_tokens = _integer(quantum_tokens, 'quantum_tokens', 1, 1024)
        self.vocab_size = _integer(model.config.get('vocab_size'), 'vocab_size', 1)
        self.context_limit = min(model.options.max_context, model.config['max_position_embeddings'])
        self.jobs = {}
        self.sequence = 0
        self.active_id = None
        self.cache = None
        self.closed = False
        self.transcript_reserved_bytes = 0
        self.journal_reserved_bytes = 0
        self.journal_written_bytes = 0
        self.state_dir = Path(state_dir).resolve() if state_dir is not None else None
        self._directory_lock = None
        self.identity = self._model_identity()
        if self.state_dir is not None:
            try:
                self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                import fcntl
                fd = os.open(self.state_dir/'.engine.lock', os.O_RDWR | os.O_CREAT, 0o600)
                self._directory_lock = os.fdopen(fd, 'a+b')
                fcntl.flock(self._directory_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._remove_stale_temporaries()
                self._restore()
            except BaseException:
                if self._directory_lock is not None:
                    self._directory_lock.close()
                raise

    def _model_identity(self):
        fingerprint = getattr(getattr(self.model, 'checkpoint', None), 'fingerprint', None)
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError('The local checkpoint must have a stable fingerprint')
        opts = self.model.options
        torch_version = None
        if self.model.config.get('model_type') == 'glm_moe_dsa':
            try:
                torch_version = metadata.version('torch')
            except metadata.PackageNotFoundError:
                pass
        # Runtime identity deliberately includes numerical scheduling and dtype.
        # Checkpoint identity is metadata-based in existing readers, not a full
        # payload hash; replay additionally checks every committed greedy token.
        payload = {'checkpoint_fingerprint': fingerprint,
                   'config': self.model.config,
                   'model_class': type(self.model).__module__+'.'+type(self.model).__qualname__,
                   'device': getattr(self.model, 'device', None),
                   'mlx_version': getattr(getattr(self.model, 'mx', None), '__version__', None),
                   'torch_distribution_version_for_glm_ties': torch_version,
                   'mlx_enable_tf32_environment': os.environ.get('MLX_ENABLE_TF32', '1'),
                   'options': {k: getattr(opts, k, None) for k in
                               ('dtype', 'prefill_chunk', 'row_chunk', 'quantized_dir', 'lossless_dir')}}
        return sha256(_canonical(payload)).hexdigest()

    def _check_owner(self):
        if get_ident() != self.owner:
            raise RuntimeError('AgentEngine must run on its model-owning thread')
        if self.closed:
            raise RuntimeError('AgentEngine is closed')

    @staticmethod
    def _reservations(job):
        tokens = len(job.input_ids)+job.max_tokens+len(job.eos_token_ids)
        # Two journal copies cover atomic replacement's old file plus temporary.
        return 4096+96*tokens, 2*(8192+16*tokens)

    def _admit(self, job):
        transcript, journal = self._reservations(job)
        if len(self.jobs) >= self.max_requests:
            raise MemoryError('Agent request capacity exhausted; forget retained jobs')
        if self.transcript_reserved_bytes+transcript > self.max_transcript_bytes:
            raise MemoryError('Agent transcript memory reservation exhausted')
        if self.state_dir is not None and self.journal_reserved_bytes+journal > self.max_journal_bytes:
            raise MemoryError('Agent journal disk reservation exhausted')
        self.transcript_reserved_bytes += transcript
        self.journal_reserved_bytes += journal if self.state_dir is not None else 0
        self.jobs[job.id] = job

    def _lookup(self, job_id):
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id) or job_id not in self.jobs:
            raise KeyError('Unknown local agent job')
        return self.jobs[job_id]

    def submit(self, input_ids, *, max_tokens=32, priority=0, agent_id='default', eos_token_id=None):
        self._check_owner()
        if not isinstance(input_ids, (list, tuple)) or not 1 <= len(input_ids) < self.context_limit:
            raise ValueError('input_ids exceeds the configured context limit or is empty')
        ids = _tokens(input_ids, self.vocab_size, 'input_ids', nonempty=True)
        count = _integer(max_tokens, 'max_tokens', 1)
        priority = _integer(priority, 'priority', -1000, 1000)
        if not isinstance(agent_id, str) or not 1 <= len(agent_id) <= 128:
            raise ValueError('agent_id must contain 1 to 128 characters')
        try:
            agent_id.encode('utf-8')
        except UnicodeError as error:
            raise ValueError('agent_id must be valid UTF-8 text') from error
        if len(ids)+count > self.context_limit:
            raise MemoryError('Agent prompt plus output exceeds the configured context limit')
        eos = self.model.config.get('eos_token_id') if eos_token_id is None else eos_token_id
        eos = [] if eos is None else [eos] if type(eos) is int else eos
        if not isinstance(eos, (list, tuple)) or len(eos) > 256:
            raise ValueError('eos_token_id must contain at most 256 token IDs')
        eos = sorted(set(_tokens(eos, self.vocab_size, 'eos_token_id')))
        self.sequence += 1
        job = AgentJob(uuid4().hex, agent_id, ids, count, priority, eos, self.sequence)
        self._admit(job)
        try:
            self._persist(job)
        except BaseException:
            self._remove(job)
            raise
        return job.id

    def _drop_session(self):
        # Evaluation is synchronous in the existing forward path. Releasing the
        # sole owner discards recurrent state without pretending truncate works.
        self.cache = None
        self.active_id = None

    def _predict(self, ids):
        output = self.model.forward([ids], past_key_values=self.cache, logits_to_keep=1)
        self.cache = output.past_key_values
        if self.cache is None:
            raise RuntimeError('The native model did not return its architecture session')
        mx = self.model.mx
        with mx.stream(self.model.stream):
            value = int(mx.argmax(output.logits[0, -1], axis=-1).item())
        if not 0 <= value < self.vocab_size:
            raise RuntimeError('The native model returned an invalid greedy token')
        return value

    def _activate(self, job):
        if self.active_id == job.id:
            return
        self._drop_session()
        self.active_id = job.id
        if not job.output_ids:
            return
        begin = perf_counter()
        try:
            # Keep exactly the ordinary-decode call boundaries: never batch an
            # old generated suffix into prefill, including on recurrent models.
            candidate = self._predict(job.input_ids)
            job.metrics['replayed_tokens'] += len(job.input_ids)
            for index, known in enumerate(job.output_ids):
                if candidate != known:
                    raise RuntimeError('replay_token_mismatch')
                if index+1 < len(job.output_ids):
                    candidate = self._predict([known])
                    job.metrics['replayed_tokens'] += 1
        finally:
            job.metrics['replay_seconds'] += perf_counter()-begin

    def step(self, max_tokens=None):
        """Run one bounded token quantum, returning its public job snapshot.

        Only durable quantum results become visible to callers. On failure the
        uncommitted quantum is discarded; retry/resume reconstructs the last
        committed transcript. Failed jobs do not block unrelated queued jobs.
        """
        self._check_owner()
        count = self.quantum_tokens if max_tokens is None else _integer(max_tokens, 'max_tokens', 1, 1024)
        ready = [job for job in self.jobs.values() if job.state == 'queued']
        if not ready:
            return None
        job = min(ready, key=lambda item: (-item.priority, item.sequence))
        previous_length = len(job.output_ids)
        previous_metrics = dict(job.metrics)
        begin = perf_counter()
        counters = self._io_counters()
        job.state = 'running'
        try:
            with self.model.lock:
                self._activate(job)
                for _ in range(min(count, job.max_tokens-len(job.output_ids))):
                    ids = job.input_ids if not job.output_ids else [job.output_ids[-1]]
                    token = self._predict(ids)
                    job.output_ids.append(token)
                    if job.metrics['first_token_latency_seconds'] is None:
                        job.metrics['first_token_latency_seconds'] = max(0., time()-job.created_at)
                    job.metrics['resident_context_peak_bytes'] = max(
                        job.metrics['resident_context_peak_bytes'],
                        int(getattr(self.cache, 'peak_bytes', getattr(self.cache, 'bytes', 0))))
                    if token in job.eos_token_ids or len(job.output_ids) == job.max_tokens:
                        job.state = 'completed'
                        job.metrics['completion_latency_seconds'] = max(0., time()-job.created_at)
                        break
            if job.state != 'completed':
                job.state = 'queued'
                self.sequence += 1
                job.sequence = self.sequence
            job.updated_at = time()
            job.metrics['inference_seconds'] += perf_counter()-begin
            job.metrics['quanta'] += 1
            after = self._io_counters()
            for name in counters:
                job.metrics[name] += max(0, after[name]-counters[name])
            self._persist(job)
            if job.state == 'completed':
                self._drop_session()
        except Exception as error:
            discarded = len(job.output_ids)-previous_length
            del job.output_ids[previous_length:]
            # Count failed/replayed work even though its output was rolled back.
            attempted = job.metrics
            job.metrics = dict(previous_metrics)
            for name in ('replay_seconds', 'replayed_tokens', 'resident_context_peak_bytes'):
                job.metrics[name] = attempted[name]
            job.metrics['inference_seconds'] += perf_counter()-begin
            job.metrics['failed_quanta'] += 1
            job.metrics['discarded_tokens'] += discarded
            after = self._io_counters()
            for name in counters:
                job.metrics[name] = previous_metrics[name]+max(0, after[name]-counters[name])
            job.state = 'failed'
            job.error = ('replay_token_mismatch' if str(error) == 'replay_token_mismatch'
                         else type(error).__name__)
            job.updated_at = time()
            self._drop_session()
            # Failure diagnostics contain class/code only, never paths/prompts.
            try:
                self._persist(job)
            except Exception:
                job.error = 'journal_write_failed'
        return self.snapshot(job.id)

    def _io_counters(self):
        stats = getattr(getattr(self.model, 'checkpoint', None), 'stats', {})
        # Existing reader transport counters are encoded file bytes, not a
        # claim about physical external-SSD traffic or OS page-cache behavior.
        from .storage import encoded_read_bytes
        checkpoint = getattr(self.model, 'checkpoint', None)
        return {'logical_bytes': int(stats.get('logical_bytes', 0)),
                'encoded_file_bytes': int(encoded_read_bytes(checkpoint))}

    def _change_state(self, job_id, state):
        self._check_owner()
        job = self._lookup(job_id)
        previous = (job.state, job.sequence, job.error, job.updated_at)
        if state == 'queued':
            if job.state not in ('paused', 'cancelled', 'failed'):
                raise ValueError('Only a paused, cancelled or failed job can resume')
            if len(job.output_ids) >= job.max_tokens or (job.output_ids and job.output_ids[-1] in job.eos_token_ids):
                raise ValueError('A complete transcript cannot resume')
            self.sequence += 1
            job.sequence = self.sequence
            job.error = None
        elif job.state not in ('queued', 'paused'):
            raise ValueError('Only a queued or paused job can pause/cancel')
        job.state = state
        job.updated_at = time()
        try:
            self._persist(job)
        except BaseException:
            job.state, job.sequence, job.error, job.updated_at = previous
            raise
        if self.active_id == job.id:
            self._drop_session()
        return self.snapshot(job.id)

    def pause(self, job_id):
        return self._change_state(job_id, 'paused')

    def cancel(self, job_id):
        return self._change_state(job_id, 'cancelled')

    def resume(self, job_id):
        return self._change_state(job_id, 'queued')

    def snapshot(self, job_id):
        self._check_owner()
        job = self._lookup(job_id)
        return {'id': job.id, 'agent_id': job.agent_id, 'state': job.state,
                'priority': job.priority, 'input_tokens': len(job.input_ids),
                'max_tokens': job.max_tokens, 'output_ids': list(job.output_ids),
                'created_at': job.created_at, 'updated_at': job.updated_at,
                'metrics': dict(job.metrics), 'error': job.error}

    def status(self):
        self._check_owner()
        return {'model_identity': self.identity, 'requests': len(self.jobs),
                'queued': sum(job.state == 'queued' for job in self.jobs.values()),
                'active_contexts': int(self.cache is not None),
                'max_resident_contexts': 1,
                'transcript_reserved_bytes': self.transcript_reserved_bytes,
                'max_transcript_bytes': self.max_transcript_bytes,
                'journal_reserved_bytes': self.journal_reserved_bytes,
                'max_journal_bytes': self.max_journal_bytes,
                'journal_written_bytes': self.journal_written_bytes,
                'max_journal_write_bytes': self.max_journal_write_bytes,
                'quantum_tokens': self.quantum_tokens,
                'scheduling': 'priority_then_round_robin; greedy only; replay on context switch'}

    def _record(self, job):
        return {'version': self.VERSION, 'model_identity': self.identity,
                'job': {key: getattr(job, key) for key in AgentJob.__dataclass_fields__}}

    def _persist(self, job):
        if self.state_dir is None:
            return
        payload = self._record(job)
        raw = _canonical({'payload': payload, 'sha256': sha256(_canonical(payload)).hexdigest()})
        if len(raw) > self._reservations(job)[1]//2:
            raise MemoryError('Job journal exceeded its disk reservation')
        if self.journal_written_bytes+len(raw) > self.max_journal_write_bytes:
            raise MemoryError('Agent cumulative journal write budget exhausted')
        fd, tmp = tempfile.mkstemp(prefix='.job-', suffix='.tmp', dir=self.state_dir)
        try:
            with os.fdopen(fd, 'wb') as stream:
                # Charge bytes offered to the writer even if a later IO/fsync
                # fails. This is logical journal traffic, not physical NAND IO.
                self.journal_written_bytes += len(raw)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.state_dir/(job.id+'.json'))
            self._fsync_directory()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _fsync_directory(self):
        fd = os.open(self.state_dir, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _remove_stale_temporaries(self):
        # A killed writer can leave only an unpublished temporary. Remove it
        # while holding the exclusive process lock; never interpret it as work.
        total = 0
        paths = list(self.state_dir.glob('.job-*.tmp'))
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError('Agent temporary journals must be regular files')
            total += path.stat().st_size
            if total > self.max_journal_bytes:
                raise MemoryError('Stale agent journals exceed the disk budget')
        for path in paths:
            path.unlink()
        if paths:
            self._fsync_directory()

    def _restore(self):
        files = [path for path in self.state_dir.glob('*.json') if _JOB_ID.fullmatch(path.stem)]
        if len(files) > self.max_requests:
            raise MemoryError('Persisted agent jobs exceed max_requests')
        total = 0
        for path in sorted(files):
            if path.is_symlink() or not path.is_file():
                raise ValueError('Agent journals must be regular files')
            size = path.stat().st_size
            total += size
            if size > 8192+16*(self.context_limit+256) or total > self.max_journal_bytes:
                raise MemoryError('Persisted agent journal size exceeds budget')
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or set(value) != {'payload', 'sha256'}:
                raise ValueError('Invalid agent journal envelope')
            payload = value['payload']
            if sha256(_canonical(payload)).hexdigest() != value['sha256']:
                raise ValueError('Agent journal checksum mismatch')
            if (not isinstance(payload, dict) or set(payload) != {'version', 'model_identity', 'job'}
                    or payload['version'] != self.VERSION or payload['model_identity'] != self.identity):
                raise ValueError('Agent journal model/runtime identity mismatch')
            record = payload['job']
            if not isinstance(record, dict) or set(record) != set(AgentJob.__dataclass_fields__):
                raise ValueError('Invalid agent job record')
            job = AgentJob(**record)
            self._validate_restored(job, path.stem)
            # Restart never silently starts recovered work. Explicit resume
            # reconstructs context and revalidates every committed token.
            if job.state in ('queued', 'running'):
                job.state = 'paused'
            self._admit(job)
            self.sequence = max(self.sequence, job.sequence)

    def _validate_restored(self, job, filename_id):
        if job.id != filename_id or not _JOB_ID.fullmatch(job.id):
            raise ValueError('Agent journal ID mismatch')
        _tokens(job.input_ids, self.vocab_size, 'input_ids', nonempty=True)
        _tokens(job.output_ids, self.vocab_size, 'output_ids')
        if not isinstance(job.eos_token_ids, list) or len(job.eos_token_ids) > 256:
            raise ValueError('Persisted EOS list exceeds its limit')
        _tokens(job.eos_token_ids, self.vocab_size, 'eos_token_ids')
        _integer(job.max_tokens, 'max_tokens', 1)
        _integer(job.priority, 'priority', -1000, 1000)
        _integer(job.sequence, 'sequence', 0)
        if (not isinstance(job.agent_id, str) or not 1 <= len(job.agent_id) <= 128
                or job.state not in _STATES or len(job.output_ids) > job.max_tokens
                or len(job.input_ids)+job.max_tokens > self.context_limit
                or any(token in job.eos_token_ids for token in job.output_ids[:-1])):
            raise ValueError('Invalid persisted agent job semantics')
        try:
            job.agent_id.encode('utf-8')
        except UnicodeError as error:
            raise ValueError('Invalid persisted agent text') from error
        complete = len(job.output_ids) == job.max_tokens or (job.output_ids and job.output_ids[-1] in job.eos_token_ids)
        if bool(complete) != (job.state == 'completed'):
            raise ValueError('Persisted completion state does not match its tokens')
        defaults = AgentJob.__dataclass_fields__['metrics'].default_factory()
        if not isinstance(job.metrics, dict) or set(job.metrics) != set(defaults):
            raise ValueError('Invalid persisted agent metrics')
        for timestamp in (job.created_at, job.updated_at):
            if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError('Invalid persisted timestamp')
        for name in ('replayed_tokens', 'logical_bytes', 'encoded_file_bytes', 'quanta',
                     'resident_context_peak_bytes', 'failed_quanta', 'discarded_tokens'):
            _integer(job.metrics[name], name, 0)
        for value in job.metrics.values():
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError('Invalid persisted timing/counter')
        if job.error is not None and (not isinstance(job.error, str) or len(job.error) > 128):
            raise ValueError('Invalid persisted error code')

    def _remove(self, job):
        transcript, journal = self._reservations(job)
        self.jobs.pop(job.id)
        self.transcript_reserved_bytes -= transcript
        self.journal_reserved_bytes -= journal if self.state_dir is not None else 0

    def forget(self, job_id):
        self._check_owner()
        job = self._lookup(job_id)
        if job.state not in _TERMINAL | {'paused'}:
            raise ValueError('Pause, cancel or finish a request before forgetting it')
        if self.state_dir is not None:
            path = self.state_dir/(job.id+'.json')
            path.unlink(missing_ok=True)
            self._fsync_directory()
        if self.active_id == job.id:
            self._drop_session()
        self._remove(job)

    def close(self):
        if get_ident() != self.owner:
            raise RuntimeError('AgentEngine must close on its model-owning thread')
        if self.closed:
            return
        self._drop_session()
        self.closed = True
        if self._directory_lock is not None:
            self._directory_lock.close()

    def __enter__(self):
        self._check_owner()
        return self

    def __exit__(self, *args):
        self.close()
