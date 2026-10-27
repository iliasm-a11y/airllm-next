"""Temporary host-wall instrumentation and exact routed-ID capture.

The observer adds no MLX evaluation/synchronization commands. It wraps commands
already issued by the engine. Reading selected router IDs has host conversion
cost and may wait for pending work; use identical observer settings for paired
runs. Timings are diagnostic host wall times, never pure Metal kernel time or
physical SSD service time. One observer may be active per Python process.
"""
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
import hashlib
import importlib
import json
from threading import Lock, local
from time import perf_counter_ns


_ACTIVE = Lock()
_NO_CONTEXT = nullcontext()
_MISSING = object()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


class _CallableProxy:
    """Forward attributes to the original mixer/MLP while observing its call."""
    def __init__(self, target, observer, kind, layer):
        self._target = target
        self._observer = observer
        self._kind = kind
        self._layer = layer

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __call__(self, *args, **kwargs):
        observer = self._observer
        state = observer._state()
        if self._kind == 'attention':
            cache = args[1] if len(args) > 1 else kwargs.get('cache')
            if cache is None or not hasattr(cache, 'position'):
                raise ValueError('Observer cannot identify mixer cache position')
            state['positions'][self._layer] = int(cache.position)
        previous = state.get('mlp_layer')
        if self._kind == 'moe':
            if self._layer not in state['positions']:
                raise ValueError('Observer saw an MLP before its mixer position')
            state['mlp_layer'] = self._layer
        try:
            with observer._span('layer.' + self._kind, self._kind, layer=self._layer,
                                position=state['positions'].get(self._layer)):
                return observer._invoke('layer.' + self._kind, self._target, args, kwargs)
        finally:
            if self._kind == 'moe':
                state['mlp_layer'] = previous


class RuntimeObserver:
    def __init__(self, model, trace=None):
        self.model = model
        self.trace = trace
        self.phase = 'prefill'
        self._routing = {}
        self._totals = defaultdict(lambda: dict(calls=0, wall_ns=0, returned_bytes=0, errors=0))
        self._phase_totals = defaultdict(lambda: defaultdict(lambda: dict(
            calls=0, wall_ns=0, returned_bytes=0, errors=0)))
        self._patches = []
        self._local = local()
        self._entered = False
        self._mutex = Lock()

    def _state(self):
        if not hasattr(self._local, 'state'):
            self._local.state = dict(positions={}, mlp_layer=None)
        return self._local.state

    def _span(self, name, category, **metadata):
        if self.trace is None:
            return _NO_CONTEXT
        return self.trace.span(name, category=category, phase=self.phase, **metadata)

    def _record(self, name, elapsed, error, returned_bytes=0):
        with self._mutex:
            for total in (self._totals[name], self._phase_totals[self.phase][name]):
                total['calls'] += 1
                total['wall_ns'] += elapsed
                total['returned_bytes'] += returned_bytes
                total['errors'] += int(error)
        if self.trace is not None and returned_bytes:
            self.trace.count(name + '.returned_bytes', returned_bytes)

    def _invoke(self, name, function, args, kwargs, *, byte_result=False):
        begin = perf_counter_ns()
        error = True
        returned_bytes = 0
        try:
            result = function(*args, **kwargs)
            error = False
            if byte_result:
                returned_bytes = result if isinstance(result, int) else len(result)
                returned_bytes = max(0, returned_bytes)
            return result
        finally:
            self._record(name, perf_counter_ns() - begin, error, returned_bytes)

    def _patch(self, target, name, value):
        original = getattr(target, name)
        namespace = getattr(target, '__dict__', {})
        previous = namespace.get(name, _MISSING)
        setattr(target, name, value)
        self._patches.append((target, name, previous, original))

    def _wrap(self, target, name, category, *, byte_result=False):
        if not hasattr(target, name):
            return
        original = getattr(target, name)

        @wraps(original)
        def observed(*args, **kwargs):
            with self._span(name, category):
                return self._invoke(name, original, args, kwargs, byte_result=byte_result)
        self._patch(target, name, observed)

    def _route_wrapper(self, original):
        @wraps(original)
        def observed(*args, **kwargs):
            with self._span('qwen_route', 'routing'):
                selected, scores = self._invoke('qwen_route', original, args, kwargs)
                # MLX supports tolist(); the fallback also supports adapters
                # returning other array types. No router values are modified.
                with self._span('route_id_host_capture', 'host_copy'):
                    if hasattr(selected, 'tolist'):
                        rows = selected.tolist()
                    else:
                        import numpy as np
                        rows = np.asarray(selected).tolist()
                state = self._state()
                layer = state.get('mlp_layer')
                if layer is None or layer not in state['positions']:
                    raise ValueError('Observed Qwen route has no owning MLP/cache position')
                x = args[1] if len(args) > 1 else kwargs.get('x')
                if x is None or len(x.shape) != 3 or int(x.shape[0]) != 1:
                    raise ValueError('Route observer requires batch-one rank-three input')
                if len(rows) != int(x.shape[1]) or any(not isinstance(row, list) for row in rows):
                    raise ValueError('Router IDs do not match the observed token block')
                start = state['positions'][layer]
                with self._mutex:
                    bank = self._routing.setdefault(str(layer), {})
                    for offset, row in enumerate(rows):
                        key = str(start + offset)
                        values = [int(expert) for expert in row]
                        if key in bank and bank[key] != values:
                            raise ValueError('Conflicting route IDs at one absolute position')
                        bank[key] = values
                return selected, scores
        return observed

    def __enter__(self):
        if self._entered or not _ACTIVE.acquire(blocking=False):
            raise RuntimeError('A RuntimeObserver is already active in this process')
        self._entered = True
        try:
            module = importlib.import_module(type(self.model).__module__)
            packed = importlib.import_module(module.__package__ + '.gguf_packed_io')
            reader = self.model.reader
            for name, category in (
                ('_packed_tile', 'weight_read_prepare'), ('_tile', 'weight_read_prepare'),
                ('get', 'small_weight_prepare'), ('embedding', 'embedding'),
                ('linear', 'weight_projection'), ('linear_experts', 'expert_projection'),
                ('linear_expert_pair', 'expert_projection'), ('output', 'output_projection'),
                ('drain', 'host_wait')):
                self._wrap(reader, name, category)
            self._wrap(self.model.checkpoint, '_pread', 'pread_host_wall', byte_result=True)
            self._wrap(packed, '_read_into_once', 'positional_syscall_host_wall', byte_result=True)
            self._wrap(self.model.mx, 'eval', 'host_wait')
            self._wrap(self.model.mx, 'synchronize', 'host_wait')
            for name, kind in (('mixers', 'attention'), ('mlps', 'moe')):
                originals = getattr(self.model, name)
                self._patch(self.model, name, [_CallableProxy(value, self, kind, layer)
                                              for layer, value in enumerate(originals)])
            self._patch(module, 'qwen_route', self._route_wrapper(module.qwen_route))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        failures = []
        for target, name, previous, original in reversed(self._patches):
            try:
                if previous is _MISSING:
                    delattr(target, name)
                else:
                    setattr(target, name, previous)
            except BaseException as error:
                failures.append(error)
        self._patches.clear()
        if self._entered:
            self._entered = False
            _ACTIVE.release()
        if failures:
            raise RuntimeError('Failed to restore RuntimeObserver patches') from failures[0]
        return False

    @property
    def routing(self):
        with self._mutex:
            return {layer: {position: list(ids) for position, ids in bank.items()}
                    for layer, bank in self._routing.items()}

    def routing_sha256(self):
        return hashlib.sha256(_canonical(self.routing)).hexdigest()

    def summary(self):
        with self._mutex:
            totals = {name: dict(value) for name, value in self._totals.items()}
            phases = {phase: {name: dict(value) for name, value in methods.items()}
                      for phase, methods in self._phase_totals.items()}
        return dict(
            methods=totals, phases=phases, routing_sha256=self.routing_sha256(),
            route_positions=sum(len(bank) for bank in self.routing.values()),
            attribution=dict(
                timing='host wall nanoseconds; nested method totals overlap and must not be added',
                _pread='includes descriptor lock and read validation',
                _read_into_once='positional syscall wrapper wall time; no physical SSD attribution',
                eval_synchronize='existing host waits include pending GPU work; not pure kernel time',
                route_capture='selected IDs copied to host; overhead shared by paired variants',
                adds_synchronization_commands=False, pure_gpu_time=False, gpu_utilization=False))

    metrics = summary
