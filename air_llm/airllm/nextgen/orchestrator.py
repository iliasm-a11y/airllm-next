"""Generic, bounded logical agents over exactly one shared local backend.

The backend owns a model, not the agent registry. Native AirLLM adapters expose
one inference lane; parallel lanes require an explicit backend capability.
Every quantum releases resident sessions. Durable transcripts, never KV arrays,
are replayed after suspension. Tool execution is delegated to an allowlist broker.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
from threading import Event, RLock, Thread, get_ident
from time import monotonic, time
from uuid import uuid4

from .experiment_guard import system_snapshot

MiB = 1024**2
GiB = 1024**3
_NAME = re.compile(r'^[A-Za-z0-9_.-]{1,128}$')
_TERMINAL = {'completed', 'failed', 'cancelled'}
_STATES = _TERMINAL | {'queued', 'running', 'paused'}


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} is outside its integer bounds')
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _name(value, kind='agent_id'):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f'{kind} must contain 1..128 ASCII letters, digits, dots, underscores or hyphens')
    return value


def _tokens(values, *, nonempty=False):
    if not isinstance(values, (list, tuple)) or (nonempty and not values):
        raise ValueError('Expected a token list')
    if any(type(token) is not int or not 0 <= token < 2**31 for token in values):
        raise ValueError('Invalid token ID')
    return list(values)


def _metrics(value, maximum=32):
    # Flat, bounded scalar telemetry has a conservative allowance in every
    # transcript reservation. Arbitrary nested Python graphs are not admitted.
    if not isinstance(value, dict) or len(value) > maximum:
        raise ValueError('Backend metrics must be a bounded flat dictionary')
    for key, item in value.items():
        if not isinstance(key, str) or len(key) > 64 or not key.isascii():
            raise ValueError('Backend metric names exceed their bounds')
        if item is None or type(item) is bool:
            continue
        if type(item) in (int, float) and math.isfinite(item) and abs(item) <= 2**63-1:
            continue
        if isinstance(item, str) and len(item) <= 128 and item.isascii():
            continue
        raise ValueError('Backend metric values must be bounded JSON scalars')
    return dict(value)


@dataclass(frozen=True)
class InferenceRequest:
    job_id: str
    agent_id: str
    input_ids: tuple
    output_ids: tuple
    max_tokens: int
    eos_token_ids: tuple = ()
    stop_event: Event = field(default_factory=Event, compare=False, repr=False)


@dataclass(frozen=True)
class InferenceResult:
    token_ids: list
    finished: bool = False
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestratorConfig:
    max_agents: int = 4096
    max_jobs: int = 4096
    max_context_tokens: int = 32768
    max_transcript_bytes: int = 64 * MiB
    max_state_bytes: int = 64 * MiB
    max_journal_write_bytes: int = 2 * GiB
    kv_budget_bytes: int = 192 * MiB
    max_concurrency: int = 1
    quantum_tokens: int = 8
    aging_seconds: float = 1.
    healthy_steps_to_grow: int = 3
    reserve_bytes: int = 3 * GiB
    swap_growth_bytes: int = 128 * MiB
    rss_bytes: int = 2 * GiB
    admitted_payload_bytes: int = 768 * MiB
    sample_interval_seconds: float = .05

    def __post_init__(self):
        for name in ('max_agents', 'max_jobs', 'max_context_tokens', 'max_transcript_bytes',
                     'max_state_bytes', 'max_journal_write_bytes', 'kv_budget_bytes',
                     'max_concurrency', 'healthy_steps_to_grow', 'rss_bytes'):
            _integer(getattr(self, name), name, 1)
        for name in ('reserve_bytes', 'swap_growth_bytes', 'admitted_payload_bytes'):
            _integer(getattr(self, name), name)
        _integer(self.quantum_tokens, 'quantum_tokens', 1, 1024)
        for name in ('aging_seconds', 'sample_interval_seconds'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.sample_interval_seconds > 1:
            raise ValueError('Memory sampling interval must be at most one second')
        if self.admitted_payload_bytes > self.rss_bytes:
            raise ValueError('Admitted payload exceeds RSS ceiling')


class LocalOrchestrator:
    """Owner-thread API; backend calls may use explicitly supported lanes.

    Registration and submission are separate. Token/job quotas are cumulative
    admissions per registered identity and survive restart/cancellation/forget.
    Retained transcripts reserve their complete possible output before dispatch.
    Priority uses integer bands plus wait-time aging, then round-robin sequence.
    Pressure is latched; clear_pressure() and explicit resume are both required.
    In-flight forwards are only cooperatively interruptible between tokens. Use
    experiment_guard.supervise for a hard owned-process RSS/swap/timeout stop.
    """
    VERSION = 1

    def __init__(self, backend, state_dir, config=None, snapshot_provider=None):
        self.backend = backend
        self.config = config or OrchestratorConfig()
        self.owner = get_ident()
        self._lock = RLock()
        self._closed = False
        self._in_step = False
        self._stop_event = Event()
        self._running = set()
        self._leases = set()
        self._release_failed = False
        self._durability_failed = False
        self._tool_calls = 0
        self._active_tools = 0
        self._tool_agents = set()
        self._active_inferences = 0
        self._peak_inferences = 0
        self._adaptive = 1
        self._healthy_steps = 0
        self._sequence = 0
        self.agents = {}
        self.jobs = {}
        self.stopped_reason = None
        self._transcript_bytes = 0
        self._journal_written = 0
        self._snapshot_provider = snapshot_provider or self._host_snapshot
        identity = getattr(backend, 'identity', None)
        if not isinstance(identity, str) or not 1 <= len(identity) <= 4096:
            raise ValueError('Backend must expose a stable identity')
        self.identity = identity
        self._backend_capacity = _integer(getattr(backend, 'max_concurrency', 1), 'backend max_concurrency', 1)
        self._kv_per_lane = _integer(getattr(backend, 'kv_bytes_per_session', None), 'backend kv_bytes_per_session', 1)
        if self._backend_capacity > 1 and getattr(backend, 'concurrent_safe', False) is not True:
            raise ValueError('Parallel backend lanes require concurrent_safe=True')
        if not callable(getattr(backend, 'advance', None)) or not callable(getattr(backend, 'release', None)):
            raise ValueError('Backend must implement advance and release')
        self._capacity = min(self._backend_capacity, self.config.max_concurrency,
                             self.config.kv_budget_bytes // self._kv_per_lane)
        if not self._capacity:
            raise MemoryError('One backend KV session exceeds the global KV budget')
        self._before = self._sample()
        self._resources = {'sample_count': 0, 'peak_rss_bytes': 0,
                           'minimum_available_bytes': self._before['available_bytes'],
                           'maximum_swap_growth_bytes': 0, 'peak_pressure_level': None}
        self._observe(self._before)
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._state_path = self.state_dir / 'orchestrator.json'
        self._directory_lock = None
        try:
            import fcntl
            fd = os.open(self.state_dir / '.orchestrator.lock', os.O_RDWR | os.O_CREAT, 0o600)
            self._directory_lock = os.fdopen(fd, 'a+b')
            fcntl.flock(self._directory_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._remove_stale_temporaries()
            self._restore()
        except BaseException:
            if self._directory_lock is not None:
                self._directory_lock.close()
            raise

    @staticmethod
    def _host_snapshot():
        import psutil
        result = system_snapshot()
        result['rss_bytes'] = psutil.Process().memory_info().rss
        return result

    def _sample(self):
        sample = dict(self._snapshot_provider())
        for name in ('available_bytes', 'swap_used_bytes', 'rss_bytes'):
            _integer(sample.get(name), name)
        pressure = sample.get('pressure_level')
        if pressure is not None and (type(pressure) is not int or pressure < 0):
            raise ValueError('Invalid pressure level')
        return sample

    def _observe(self, sample):
        self._resources['sample_count'] += 1
        self._resources['peak_rss_bytes'] = max(self._resources['peak_rss_bytes'], sample['rss_bytes'])
        self._resources['minimum_available_bytes'] = min(self._resources['minimum_available_bytes'], sample['available_bytes'])
        self._resources['maximum_swap_growth_bytes'] = max(
            self._resources['maximum_swap_growth_bytes'], sample['swap_used_bytes'] - self._before['swap_used_bytes'])
        if sample.get('pressure_level') is not None:
            self._resources['peak_pressure_level'] = max(self._resources['peak_pressure_level'] or 0, sample['pressure_level'])

    def _violation(self, sample):
        cfg = self.config
        if sample.get('pressure_level') not in (None, 1):
            return 'memory_pressure'
        if sample['available_bytes'] < cfg.reserve_bytes:
            return 'system_reserve'
        if sample['swap_used_bytes'] - self._before['swap_used_bytes'] > cfg.swap_growth_bytes:
            return 'swap_growth'
        if sample['rss_bytes'] > cfg.rss_bytes:
            return 'rss_limit'
        return None

    def _check(self, mutable=True):
        if self._closed:
            raise RuntimeError('Orchestrator is closed')
        if mutable and get_ident() != self.owner:
            raise RuntimeError('Mutations must run on the orchestrator owner thread')
        if mutable and self._in_step:
            raise RuntimeError('Cannot mutate the orchestrator during an inference quantum')

    def _request(self, job):
        return InferenceRequest(job['id'], job['agent_id'], tuple(job['input_ids']),
                                tuple(job['output_ids']), job['max_tokens'],
                                tuple(job['eos_token_ids']), self._stop_event)

    def _validate_request(self, request):
        validator = getattr(self.backend, 'validate', None)
        if validator is not None:
            validator(request)

    def register(self, agent_id, *, token_quota=4096, job_quota=16, tools=()):
        self._check()
        _name(agent_id)
        _integer(token_quota, 'token_quota', 1)
        _integer(job_quota, 'job_quota', 1)
        if not isinstance(tools, (tuple, list)) or len(tools) > 16:
            raise ValueError('Agent tool allowlist must contain at most 16 names')
        tools = sorted(set(_name(tool, 'tool name') for tool in tools))
        if sum(len(tool) for tool in tools) > 1024:
            raise ValueError('Agent tool allowlist exceeds its memory allowance')
        if agent_id in self.agents:
            raise ValueError('Agent identity is already registered')
        if len(self.agents) >= self.config.max_agents:
            raise MemoryError('Registered-agent capacity exhausted')
        if self._transcript_bytes + (len(self.agents)+1)*4096 > self.config.max_transcript_bytes:
            raise MemoryError('Registry plus transcript memory budget exhausted')
        self.agents[agent_id] = {'token_quota': token_quota, 'job_quota': job_quota,
                                 'admitted_tokens': 0, 'admitted_jobs': 0, 'tools': tools}
        try:
            self._persist()
        except BaseException:
            del self.agents[agent_id]
            raise
        return self.agent_snapshot(agent_id)

    def agent_snapshot(self, agent_id):
        self._check(False)
        return json.loads(json.dumps({'agent_id': agent_id, **self.agents[agent_id]}))

    @staticmethod
    def _reservation(job):
        return 16384 + 96*(len(job['input_ids']) + job['max_tokens'] + len(job['eos_token_ids']))

    def submit(self, agent_id, input_ids, *, max_tokens=32, priority=0, eos_token_ids=()):
        self._check()
        agent = self.agents[agent_id]
        if self.stopped_reason:
            raise RuntimeError('Orchestrator is suspended; explicitly clear pressure/resume first')
        if not isinstance(input_ids, (list, tuple)) or len(input_ids) > self.config.max_context_tokens:
            raise MemoryError('Input token list exceeds context budget')
        if not isinstance(eos_token_ids, (list, tuple)) or len(eos_token_ids) > 256:
            raise MemoryError('EOS token list exceeds its raw budget')
        ids = _tokens(input_ids, nonempty=True)
        eos = sorted(set(_tokens(eos_token_ids)))
        _integer(max_tokens, 'max_tokens', 1)
        _integer(priority, 'priority', -1000, 1000)
        if len(eos) > 256 or len(ids) + max_tokens > self.config.max_context_tokens:
            raise MemoryError('Job context or EOS budget exceeded')
        if len(self.jobs) >= self.config.max_jobs:
            raise MemoryError('Retained-job capacity exhausted; forget terminal jobs')
        if agent['admitted_jobs'] >= agent['job_quota'] or agent['admitted_tokens'] + max_tokens > agent['token_quota']:
            raise MemoryError('Agent cumulative job/token quota exhausted')
        self._sequence += 1
        now = time()
        job = {'id': uuid4().hex, 'agent_id': agent_id, 'input_ids': ids, 'output_ids': [],
               'eos_token_ids': eos, 'max_tokens': max_tokens, 'priority': priority,
               'state': 'queued', 'sequence': self._sequence, 'created_at': now,
               'queued_at': now, 'updated_at': now, 'error': None,
               'metrics': {'service_seconds': 0., 'queue_seconds': 0., 'quanta': 0,
                           'ttft_seconds': None, 'completion_seconds': None,
                           'backend_metrics': {}}}
        self._validate_request(self._request(job))
        reservation = self._reservation(job)
        if self._transcript_bytes + reservation + len(self.agents)*4096 > self.config.max_transcript_bytes:
            raise MemoryError('Registry plus transcript memory budget exhausted')
        self._transcript_bytes += reservation
        self.jobs[job['id']] = job
        agent['admitted_jobs'] += 1
        agent['admitted_tokens'] += max_tokens
        try:
            self._persist()
        except BaseException:
            self._transcript_bytes -= reservation
            del self.jobs[job['id']]
            agent['admitted_jobs'] -= 1
            agent['admitted_tokens'] -= max_tokens
            raise
        return job['id']

    def _monitor(self, done):
        while not done.wait(self.config.sample_interval_seconds):
            try:
                sample = self._sample()
                with self._lock:
                    self._observe(sample)
                    reason = self._violation(sample)
                    if reason:
                        self.stopped_reason = reason
                        self._stop_event.set()
                        return
            except Exception:
                with self._lock:
                    self.stopped_reason = 'resource_sampler_failed'
                    self._stop_event.set()
                return

    def _invoke(self, job, count):
        with self._lock:
            self._running.add(job['id'])
            self._leases.add(job['id'])
            self._active_inferences += 1
            self._peak_inferences = max(self._peak_inferences, self._active_inferences)
        begin = monotonic()
        started_at = time()
        try:
            try:
                result = self.backend.advance(self._request(job), count)
            finally:
                with self._lock:
                    self._active_inferences -= 1
            if not isinstance(result, InferenceResult):
                raise TypeError('Backend must return InferenceResult')
            if not isinstance(result.token_ids, (list, tuple)) or len(result.token_ids) > count:
                raise ValueError('Backend returned too many quantum tokens')
            tokens = _tokens(result.token_ids)
            if type(result.finished) is not bool or len(tokens) > count or not tokens and not result.finished:
                raise ValueError('Backend returned invalid quantum progress')
            if len(job['output_ids']) + len(tokens) > job['max_tokens']:
                raise ValueError('Backend exceeded output quota')
            if any(token in job['eos_token_ids'] for token in tokens[:-1]):
                raise ValueError('Backend generated tokens after EOS')
            _metrics(result.metrics, maximum=31)
            candidate = InferenceRequest(job['id'], job['agent_id'], tuple(job['input_ids']),
                                         tuple(job['output_ids'] + tokens), job['max_tokens'],
                                         tuple(job['eos_token_ids']), self._stop_event)
            self._validate_request(candidate)
            return result, None, monotonic()-begin, started_at
        except Exception as error:
            return None, type(error).__name__, monotonic()-begin, started_at
        finally:
            try:
                self.backend.release(job['id'])
            except Exception:
                with self._lock:
                    # A failed release does not prove that native arrays were
                    # dropped. Keep the lease charged and require a new owner.
                    self._release_failed = True
                    self.stopped_reason = 'backend_release_failed'
                    self._stop_event.set()
            else:
                with self._lock:
                    self._leases.discard(job['id'])
            finally:
                with self._lock:
                    self._running.discard(job['id'])

    def step(self):
        self._check()
        with self._lock:
            sample = self._sample()
            self._observe(sample)
            reason = self._violation(sample)
            if reason:
                self._suspend(reason)
                self._persist()
            if self.stopped_reason:
                return []
            ready = [job for job in self.jobs.values() if job['state'] == 'queued']
            if not ready:
                return []
            available_lanes = max(0, (sample['available_bytes'] - self.config.reserve_bytes -
                                      self.config.admitted_payload_bytes) // self._kv_per_lane)
            if not available_lanes:
                self._suspend('admission_reserve')
                self._persist()
                return []
            cap = min(self._capacity, available_lanes)
            self._adaptive = min(self._adaptive, cap)
            self._healthy_steps += 1
            if self._healthy_steps >= self.config.healthy_steps_to_grow:
                self._adaptive = min(cap, self._adaptive+1)
                self._healthy_steps = 0
            now = time()
            ready.sort(key=lambda job: (-job['priority'] - int(max(0., now-job['queued_at']) / self.config.aging_seconds),
                                       job['sequence']))
            selected = ready[:self._adaptive]
            self._in_step = True
            for job in selected:
                job['state'] = 'running'
                job['metrics']['queue_seconds'] += max(0., now-job['queued_at'])
        done = Event()
        monitor = Thread(target=self._monitor, args=(done,), daemon=True)
        monitor.start()
        try:
            if len(selected) == 1:
                results = [self._invoke(selected[0], min(self.config.quantum_tokens,
                           selected[0]['max_tokens']-len(selected[0]['output_ids'])))]
            else:
                # Exactly one backend object is shared. No model factory or
                # extra model process is created per logical agent/lane.
                with ThreadPoolExecutor(max_workers=len(selected), thread_name_prefix='local-inference') as executor:
                    futures = [executor.submit(self._invoke, job, min(self.config.quantum_tokens,
                               job['max_tokens']-len(job['output_ids']))) for job in selected]
                    results = [future.result() for future in futures]
        except BaseException:
            with self._lock:
                self._suspend('inference_interrupted')
                self._persist()
            raise
        finally:
            done.set()
            monitor.join()
            with self._lock:
                self._in_step = False
        with self._lock:
            try:
                sample = self._sample()
                self._observe(sample)
                reason = self._violation(sample)
                if reason:
                    self._suspend(reason)
            except Exception:
                self._suspend('resource_sampler_failed')
            for job, (result, error, seconds, started_at) in zip(selected, results):
                now = time()
                job['updated_at'] = now
                job['metrics']['service_seconds'] += seconds
                job['metrics']['quanta'] += 1
                if self.stopped_reason:
                    job['state'] = 'paused'
                elif error:
                    job['state'] = 'failed'
                    job['error'] = error
                else:
                    job['output_ids'].extend(result.token_ids)
                    job['metrics']['backend_metrics'] = dict(result.metrics)
                    if result.token_ids and job['metrics']['ttft_seconds'] is None:
                        first_seconds = result.metrics.get('first_token_seconds')
                        # Without a token callback/metric this is the first
                        # completed quantum upper bound, explicitly labelled.
                        exact = type(first_seconds) in (int, float) and 0 <= first_seconds <= seconds
                        job['metrics']['ttft_seconds'] = max(0., started_at-job['created_at']+first_seconds) if exact else max(0., started_at-job['created_at']+seconds)
                        job['metrics']['backend_metrics']['ttft_is_quantum_upper_bound'] = not exact
                    complete = (result.finished or len(job['output_ids']) == job['max_tokens'] or
                                bool(job['output_ids'] and job['output_ids'][-1] in job['eos_token_ids']))
                    job['state'] = 'completed' if complete else 'queued'
                    if complete:
                        job['metrics']['completion_seconds'] = max(0., now-job['created_at'])
                    else:
                        self._sequence += 1
                        job['sequence'] = self._sequence
                        job['queued_at'] = now
            if self.stopped_reason:
                self._suspend(self.stopped_reason)
            try:
                self._persist()
            except BaseException:
                # Disk still contains the last complete checkpoint. Do not
                # permit inference from the uncommitted in-memory suffix.
                self._durability_failed = True
                self._suspend('journal_commit_failed')
                raise
            return [self.snapshot(job['id']) for job in selected]

    def _suspend(self, reason):
        self.stopped_reason = reason
        self._stop_event.set()
        self._adaptive = 1
        self._healthy_steps = 0
        for job in self.jobs.values():
            if job['state'] in ('queued', 'running'):
                job['state'] = 'paused'

    def suspend_all(self, reason='manual'):
        self._check()
        _name(reason, 'suspension reason')
        self._suspend(reason)
        self._persist()

    def clear_pressure(self):
        self._check()
        if self._release_failed or self._durability_failed:
            raise RuntimeError('Close and restore from the last durable checkpoint after backend/disk failure')
        sample = self._sample()
        self._observe(sample)
        reason = self._violation(sample)
        if reason or sample['available_bytes'] < self.config.reserve_bytes+self.config.admitted_payload_bytes+self._kv_per_lane:
            raise MemoryError('Resources do not permit clearing the suspension')
        self.stopped_reason = None
        self._stop_event.clear()
        self._persist()

    def execute_tool(self, agent_id, broker, tool, arguments):
        """Explicit trusted-host dispatch; model text never executes a tool.

        Both the durable agent grant and the broker's independent allowlist /
        quota must authorize the operation. Tools have their own process and
        resource limits; they do not occupy an inference lane or model copy.
        """
        self._check()
        agent = self.agents[agent_id]
        if tool not in agent['tools']:
            raise PermissionError('Tool is outside the registered agent grant')
        sample = self._sample()
        self._observe(sample)
        reason = self._violation(sample)
        if reason:
            self._suspend(reason)
            self._persist()
        if self.stopped_reason:
            raise RuntimeError('Tools are blocked while the orchestrator is suspended')
        self._in_step = True
        self._active_tools += 1
        self._tool_agents.add(agent_id)
        self._tool_calls += 1
        try:
            return broker.execute(agent_id, tool, arguments)
        finally:
            self._active_tools -= 1
            self._tool_agents.discard(agent_id)
            self._in_step = False

    def _change(self, job_id, state):
        self._check()
        job = self.jobs[job_id]
        old = dict(job)
        if state == 'queued':
            if self.stopped_reason:
                raise RuntimeError('Clear the suspension before explicitly resuming work')
            if job['state'] != 'paused':
                raise ValueError('Only paused jobs can resume')
            self._sequence += 1
            job['sequence'] = self._sequence
            job['queued_at'] = time()
        elif job['state'] not in ('queued', 'paused'):
            raise ValueError('Only queued/paused jobs can pause or cancel')
        job['state'] = state
        job['updated_at'] = time()
        try:
            self._persist()
        except BaseException:
            job.update(old)
            raise
        return self.snapshot(job_id)

    def pause(self, job_id):
        return self._change(job_id, 'paused')

    def resume(self, job_id):
        return self._change(job_id, 'queued')

    def cancel(self, job_id):
        return self._change(job_id, 'cancelled')

    def resume_all(self):
        self._check()
        if self.stopped_reason:
            raise RuntimeError('Clear the suspension before explicitly resuming work')
        for job in self.jobs.values():
            if job['state'] == 'paused':
                self._sequence += 1
                job['state'] = 'queued'
                job['sequence'] = self._sequence
                job['queued_at'] = time()
        self._persist()

    def forget(self, job_id):
        self._check()
        job = self.jobs[job_id]
        if job['state'] not in _TERMINAL:
            raise ValueError('Only terminal jobs can be forgotten')
        del self.jobs[job_id]
        self._transcript_bytes -= self._reservation(job)
        try:
            self._persist()
        except BaseException:
            self.jobs[job_id] = job
            self._transcript_bytes += self._reservation(job)
            raise

    def snapshot(self, job_id):
        self._check(False)
        with self._lock:
            return json.loads(json.dumps(self.jobs[job_id]))

    def status(self):
        self._check(False)
        with self._lock:
            counts = {state: sum(job['state'] == state for job in self.jobs.values()) for state in _STATES}
            active = {self.jobs[job]['agent_id'] for job in self._running} | self._tool_agents
            waiting = {job['agent_id'] for job in self.jobs.values() if job['state'] == 'queued'}
            paused = {job['agent_id'] for job in self.jobs.values() if job['state'] == 'paused'}
            return {'registered_agents': len(self.agents), 'active_agents': len(active),
                    'waiting_agents': len(waiting-active), 'paused_agents': len(paused-active),
                    'inferences_in_flight': self._active_inferences,
                    'peak_simultaneous_inferences': self._peak_inferences,
                    'adaptive_concurrency': self._adaptive, 'backend_capacity': self._backend_capacity,
                    'effective_capacity': self._capacity, 'model_instances': 1,
                    'transcript_reserved_bytes': self._transcript_bytes,
                    'registry_reserved_bytes': len(self.agents)*4096,
                    'kv_reserved_bytes': len(self._leases)*self._kv_per_lane,
                    'journal_written_bytes': self._journal_written,
                    'active_tools': self._active_tools, 'tool_calls': self._tool_calls,
                    'stopped_reason': self.stopped_reason,
                    'durability_failed': self._durability_failed,
                    'resource_metrics': dict(self._resources),
                    **{'jobs_'+state: count for state, count in counts.items()}}

    def _record(self):
        return {'version': self.VERSION, 'backend_identity': self.identity,
                'config': asdict(self.config), 'backend_capacity': self._backend_capacity,
                'kv_bytes_per_session': self._kv_per_lane, 'agents': self.agents,
                'jobs': self.jobs, 'sequence': self._sequence,
                'stopped_reason': self.stopped_reason, 'journal_written_bytes': self._journal_written}

    def _persist(self):
        try:
            self._write_snapshot()
        except BaseException:
            self._durability_failed = True
            self._suspend('journal_commit_failed')
            raise

    def _write_snapshot(self):
        payload = self._record()
        raw_payload = _json(payload)
        raw = _json({'payload': payload, 'sha256': sha256(raw_payload).hexdigest()})
        if len(raw)*2 > self.config.max_state_bytes:
            raise MemoryError('Atomic journal old+new disk budget exhausted')
        if self._journal_written + len(raw) > self.config.max_journal_write_bytes:
            self._suspend('journal_write_budget')
            raise MemoryError('Cumulative journal write budget exhausted')
        fd, temporary = tempfile.mkstemp(prefix='.state-', suffix='.tmp', dir=self.state_dir)
        try:
            with os.fdopen(fd, 'wb') as stream:
                self._journal_written += len(raw)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._state_path)
            directory = os.open(self.state_dir, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _restore(self):
        if not self._state_path.exists():
            return
        if self._state_path.is_symlink() or not self._state_path.is_file():
            raise ValueError('Journal must be a regular local file')
        if self._state_path.stat().st_size*2 > self.config.max_state_bytes:
            raise MemoryError('Persisted state exceeds disk budget')
        envelope = json.loads(self._state_path.read_text())
        if not isinstance(envelope, dict) or set(envelope) != {'payload', 'sha256'}:
            raise ValueError('Invalid journal envelope')
        payload = envelope['payload']
        if sha256(_json(payload)).hexdigest() != envelope['sha256']:
            raise ValueError('Journal checksum mismatch')
        expected = self._record()
        if not isinstance(payload, dict) or set(payload) != set(expected):
            raise ValueError('Invalid journal schema')
        for key in ('version', 'backend_identity', 'config', 'backend_capacity', 'kv_bytes_per_session'):
            if payload[key] != expected[key]:
                raise ValueError('Persisted backend identity, capabilities or limits mismatch')
        agents, jobs = payload['agents'], payload['jobs']
        if not isinstance(agents, dict) or len(agents) > self.config.max_agents:
            raise MemoryError('Persisted registry exceeds capacity')
        if not isinstance(jobs, dict) or len(jobs) > self.config.max_jobs:
            raise MemoryError('Persisted jobs exceed capacity')
        for agent_id, agent in agents.items():
            _name(agent_id)
            if not isinstance(agent, dict) or set(agent) != {'token_quota', 'job_quota', 'admitted_tokens', 'admitted_jobs', 'tools'}:
                raise ValueError('Invalid persisted agent')
            for key in ('token_quota', 'job_quota'):
                _integer(agent[key], key, 1)
            for counter, quota in (('admitted_tokens', 'token_quota'), ('admitted_jobs', 'job_quota')):
                _integer(agent[counter], counter, 0, agent[quota])
            if not isinstance(agent['tools'], list) or len(agent['tools']) > 16:
                raise ValueError('Invalid persisted tool allowlist')
            for tool in agent['tools']:
                _name(tool, 'tool name')
            if sum(len(tool) for tool in agent['tools']) > 1024:
                raise ValueError('Persisted tool allowlist exceeds memory allowance')
        reserved = 0
        retained_by_agent = {agent: [0, 0] for agent in agents}
        expected_job_keys = {'id', 'agent_id', 'input_ids', 'output_ids', 'eos_token_ids', 'max_tokens',
                             'priority', 'state', 'sequence', 'created_at', 'queued_at', 'updated_at', 'error', 'metrics'}
        for job_id, job in jobs.items():
            if not isinstance(job, dict) or set(job) != expected_job_keys or job_id != job['id']:
                raise ValueError('Invalid persisted job schema')
            if not re.fullmatch(r'[a-f0-9]{32}', job_id) or job['agent_id'] not in agents or job['state'] not in _STATES:
                raise ValueError('Invalid persisted job identity/state')
            _integer(job['max_tokens'], 'max_tokens', 1)
            _integer(job['priority'], 'priority', -1000, 1000)
            _integer(job['sequence'], 'sequence')
            _tokens(job['input_ids'], nonempty=True)
            _tokens(job['output_ids'])
            _tokens(job['eos_token_ids'])
            if len(job['eos_token_ids']) > 256 or len(job['input_ids'])+job['max_tokens'] > self.config.max_context_tokens:
                raise MemoryError('Persisted context exceeds budget')
            if len(job['output_ids']) > job['max_tokens'] or any(t in job['eos_token_ids'] for t in job['output_ids'][:-1]):
                raise ValueError('Invalid persisted output transcript')
            complete = len(job['output_ids']) == job['max_tokens'] or bool(job['output_ids'] and job['output_ids'][-1] in job['eos_token_ids'])
            if complete and job['state'] != 'completed':
                raise ValueError('Complete persisted transcript requires completed state')
            for key in ('created_at', 'queued_at', 'updated_at'):
                value = job[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid persisted timestamp')
            if job['error'] is not None and (not isinstance(job['error'], str) or len(job['error']) > 128):
                raise ValueError('Invalid persisted diagnostic')
            metrics = job['metrics']
            if not isinstance(metrics, dict) or set(metrics) != {'service_seconds', 'queue_seconds', 'quanta', 'ttft_seconds', 'completion_seconds', 'backend_metrics'}:
                raise ValueError('Invalid persisted metrics')
            for key in ('service_seconds', 'queue_seconds', 'ttft_seconds', 'completion_seconds'):
                value = metrics[key]
                if value is None and key in ('ttft_seconds', 'completion_seconds'):
                    continue
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid persisted metric value')
            _integer(metrics['quanta'], 'quanta')
            _metrics(metrics['backend_metrics'])
            self._validate_request(self._request(job))
            reserved += self._reservation(job)
            retained_by_agent[job['agent_id']][0] += 1
            retained_by_agent[job['agent_id']][1] += job['max_tokens']
            if job['state'] in ('running', 'queued'):
                job['state'] = 'paused'
        if reserved + len(agents)*4096 > self.config.max_transcript_bytes:
            raise MemoryError('Persisted registry/transcripts exceed memory budget')
        for agent_id, (retained_jobs, retained_tokens) in retained_by_agent.items():
            if agents[agent_id]['admitted_jobs'] < retained_jobs or agents[agent_id]['admitted_tokens'] < retained_tokens:
                raise ValueError('Persisted quota counters do not cover retained jobs')
        _integer(payload['sequence'], 'sequence')
        _integer(payload['journal_written_bytes'], 'journal_written_bytes', 0, self.config.max_journal_write_bytes)
        if payload['stopped_reason'] is not None:
            _name(payload['stopped_reason'], 'suspension reason')
        self.agents, self.jobs = agents, jobs
        self._transcript_bytes = reserved
        self._sequence = max([payload['sequence'], *(job['sequence'] for job in jobs.values())])
        self._journal_written = payload['journal_written_bytes'] + self._state_path.stat().st_size
        self.stopped_reason = payload['stopped_reason']
        if self.stopped_reason:
            self._stop_event.set()

    def _remove_stale_temporaries(self):
        # Exclusive lock makes abandoned atomic writes safe to remove. Bound
        # both bytes and directory entries before allocating another snapshot.
        stale, total = [], 0
        for path in self.state_dir.glob('.state-*.tmp'):
            if len(stale) >= 1024 or path.is_symlink() or not path.is_file():
                raise ValueError('Invalid or excessive stale state temporaries')
            total += path.stat().st_size
            if total > self.config.max_state_bytes:
                raise MemoryError('Stale state temporaries exceed disk budget')
            stale.append(path)
        for path in stale:
            path.unlink()
        if stale:
            fd = os.open(self.state_dir, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def close(self):
        if self._closed:
            return
        self._check()
        self._closed = True
        self._directory_lock.close()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()
