"""Optional, bounded expert prediction which never changes exact routing.

The predictor uses previous observations of *the same layer*. Learned expert
transitions take precedence over simple last-route reuse. This is probabilistic
weight lookahead, unrelated to speculative token decoding. The caller still
executes every expert selected by the real router.

All reads use the existing SingleFlight scheduler and its byte reservations.
Only futures created here are owned here; an existing exact lookahead/demand
future is never adopted. ``take`` removes failed predictions and returns None,
so the ordinary demand path can retry the actual read. Discarding a running read
waits for it: Python cannot cancel an active file read safely. Until then its
reservation remains held in both budgets.
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from numbers import Integral
from threading import Event, RLock
from time import perf_counter

from .cache import size_of


@dataclass
class _Layer:
    previous: tuple = ()
    predicted: tuple = ()


@dataclass
class _Read:
    layer: int
    expert: int
    key: object
    item: tuple
    size: int
    state: str = 'pending'
    done: Event = field(default_factory=Event)


class SpecPrefetch:
    """Bounded learned-transition/reuse predictor sharing a SingleFlight.

    ``predict(layer)`` is called before the real router is available. The caller
    may ``submit(layer, expert, key, size, loader)`` for each returned expert.
    ``observe(layer, actual_experts)`` cancels/discards incorrect predictions and
    updates the bounded history. Call ``take(key)`` before the scheduler's normal
    demand take, and ``discard()`` before a shared scheduler drain.

    ``budget_bytes`` caps decoded predictions, including completed unconsumed
    buffers; ``max_pending`` caps zero-byte futures too. Predictor metadata has
    independent global layer, transition-edge and route-length caps. Counters
    report decoded payload bytes, not physical SSD traffic or measured bandwidth.
    """
    def __init__(self, io, budget_bytes, *, max_predictions=1, max_layers=256,
                 max_transitions=4096, max_pending=32, max_route_experts=64):
        for name, value, minimum in (
                ('budget_bytes', budget_bytes, 0), ('max_predictions', max_predictions, 1),
                ('max_layers', max_layers, 1), ('max_transitions', max_transitions, 0),
                ('max_pending', max_pending, 1), ('max_route_experts', max_route_experts, 1)):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if budget_bytes > io.budget:
            raise ValueError('SpecPrefetch budget exceeds the shared IO budget')
        self.io = io
        self.budget = budget_bytes
        self.max_predictions = min(max_predictions, max_route_experts)
        self.max_layers = max_layers
        self.max_transitions = max_transitions
        self.max_pending = max_pending
        self.max_route_experts = max_route_experts
        self.layers = OrderedDict()
        self.transitions = OrderedDict()
        self.pending = {}
        self.bytes = 0
        self.peak_bytes = 0
        self.stats = Counter()
        self.closed = False
        self.lock = RLock()

    @staticmethod
    def _id(value, name):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
        return int(value)

    def _layer(self, layer):
        """Return state and any evicted layer; caller holds our lock."""
        state = self.layers.get(layer)
        evicted = None
        if state is None:
            if len(self.layers) == self.max_layers:
                evicted, _ = self.layers.popitem(last=False)
                for edge in tuple(self.transitions):
                    if edge[0] == evicted:
                        del self.transitions[edge]
                self.stats['layer_evictions'] += 1
            state = self.layers[layer] = _Layer()
        self.layers.move_to_end(layer)
        return state, evicted

    def predict(self, layer):
        start = perf_counter()
        layer = self._id(layer, 'layer')
        with self.lock:
            if self.closed:
                return ()
            state, evicted = self._layer(layer)
            scores = Counter()
            previous = set(state.previous)
            for (which, source, target), count in self.transitions.items():
                if which == layer and source in previous:
                    scores[target] += count
            if scores:
                predicted = tuple(sorted(scores, key=lambda e: (-scores[e], e))[:self.max_predictions])
                self.stats['transition_predictions'] += len(predicted)
            else:
                predicted = state.previous[:self.max_predictions]
                self.stats['reuse_predictions'] += len(predicted)
            state.predicted = predicted
            self.stats['prediction_calls'] += 1
            self.stats['prediction_seconds'] += perf_counter() - start
        if evicted is not None:
            self.discard(layer=evicted)
        return predicted

    def observe(self, layer, experts):
        """Learn observed routes only; discarded predictions cannot affect them.

        At most max_route_experts input positions are retained. Oversized prefill
        routes are explicitly counted as truncated observations; exact execution
        remains the caller's responsibility and is never pruned here.
        """
        start = perf_counter()
        layer = self._id(layer, 'layer')
        route = []
        seen = set()
        truncated = False
        for position, value in enumerate(experts):
            if position == self.max_route_experts:
                truncated = True
                break
            expert = self._id(value, 'expert')
            if expert not in seen:
                route.append(expert)
                seen.add(expert)
        with self.lock:
            if self.closed:
                return
            state, evicted = self._layer(layer)
            self.stats['prediction_hits'] += len(set(state.predicted) & seen)
            self.stats['prediction_misses'] += len(set(state.predicted) - seen)
            self.stats['observations'] += 1
            self.stats['truncated_observations'] += int(truncated)
            if self.max_transitions:
                for source in state.previous:
                    for target in route:
                        edge = (layer, source, target)
                        # Saturating counts keep integer storage bounded too.
                        self.transitions[edge] = min(65535, self.transitions.get(edge, 0) + 1)
                        self.transitions.move_to_end(edge)
                        if len(self.transitions) > self.max_transitions:
                            self.transitions.popitem(last=False)
                            self.stats['transition_evictions'] += 1
            state.previous = tuple(route)
            state.predicted = ()
            self.stats['observe_seconds'] += perf_counter() - start
        if evicted is not None:
            self.discard(layer=evicted)
        self._discard(lambda entry: entry.layer == layer and entry.expert not in seen)

    def _release_locked(self, entry, outcome, payload_bytes=0):
        if self.pending.get(entry.key) is not entry:
            return
        del self.pending[entry.key]
        self.bytes -= entry.size
        self.stats[outcome] += 1
        self.stats[outcome + '_reservation_bytes'] += entry.size
        if payload_bytes:
            self.stats[outcome + '_payload_bytes'] += payload_bytes
        entry.done.set()

    def _reap_locked(self):
        # A shared drain/take may have consumed a prediction without our wrapper.
        # Reconcile it without guessing whether the external use was useful.
        with self.io.lock:
            stale = [entry for entry in self.pending.values()
                     if entry.state == 'pending' and self.io.pending.get(entry.key) is not entry.item]
        for entry in stale:
            payload = 0
            if entry.item[0].done() and not entry.item[0].cancelled():
                try:
                    payload = size_of(entry.item[0].result())
                except BaseException:
                    self.stats['externally_released_failures'] += 1
            self._release_locked(entry, 'externally_released', payload)

    def submit(self, layer, expert, key, size, loader):
        """Submit one predicted payload, without adopting existing IO futures."""
        layer = self._id(layer, 'layer')
        expert = self._id(expert, 'expert')
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError('IO reservation must be a nonnegative integer')
        with self.lock:
            self._reap_locked()
            if self.closed or not self.budget:
                return False
            state = self.layers.get(layer)
            if state is None or expert not in state.predicted:
                self.stats['not_predicted'] += 1
                return False
            if key in self.pending:
                self.stats['deduplicated'] += 1
                return True
            if len(self.pending) >= self.max_pending or self.bytes + size > self.budget:
                self.stats['budget_rejections'] += 1
                return False
            with self.io.lock:
                if key in self.io.pending:
                    self.stats['existing_exact_reads'] += 1
                    return False
                try:
                    if not self.io.submit(key, size, loader):
                        self.stats['shared_budget_rejections'] += 1
                        return False
                except BaseException:
                    self.stats['submit_failures'] += 1
                    return False
                item = self.io.pending[key]
            self.pending[key] = _Read(layer, expert, key, item, size)
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.stats['submitted'] += 1
            return True

    def take(self, key):
        """Return an owned prediction or None; prediction errors demand a retry."""
        with self.lock:
            self._reap_locked()
            entry = self.pending.get(key)
            if entry is None:
                return None
            discarding = entry.state == 'discarding'
            if not discarding:
                entry.state = 'taking'
        if discarding:
            entry.done.wait()
            return None
        start = perf_counter()
        outcome = 'useful'
        value = None
        payload = 0
        try:
            value = entry.item[0].result()
            payload = size_of(value)
        except BaseException:
            # A prediction is never authoritative. Its failed future must be
            # removed before the normal demand path retries the same identity.
            outcome = 'failed'
            value = None
        finally:
            with self.lock:
                with self.io.lock:
                    if self.io.pending.get(key) is entry.item:
                        del self.io.pending[key]
                        self.io.bytes -= entry.size
                        self.io.stats['consumed'] += 1
                self._release_locked(entry, outcome, payload)
                self.stats['take_wait_seconds'] += perf_counter() - start
        return value

    def _discard(self, predicate):
        with self.lock:
            self._reap_locked()
            entries = []
            waiters = []
            for entry in self.pending.values():
                if not predicate(entry):
                    continue
                if entry.state == 'pending':
                    entry.state = 'discarding'
                    entries.append(entry)
                else:
                    waiters.append(entry.done)
        for entry in entries:
            start = perf_counter()
            future = entry.item[0]
            payload = 0
            outcome = 'cancelled' if future.cancel() else 'wasted'
            try:
                if not future.cancelled():
                    payload = size_of(future.result())
            except BaseException:
                outcome = 'discarded_failed'
            finally:
                with self.lock:
                    with self.io.lock:
                        if self.io.pending.get(entry.key) is entry.item:
                            del self.io.pending[entry.key]
                            self.io.bytes -= entry.size
                            self.io.stats['discarded'] += 1
                    self._release_locked(entry, outcome, payload)
                    self.stats['discard_wait_seconds'] += perf_counter() - start
        for done in waiters:
            done.wait()

    def discard(self, layer=None):
        """Release only owned predictions; read failures are always suppressed."""
        if layer is not None:
            layer = self._id(layer, 'layer')
        self._discard(lambda entry: layer is None or entry.layer == layer)

    def metrics(self):
        with self.lock:
            self._reap_locked()
            return dict(self.stats, outstanding_bytes=self.bytes, peak_bytes=self.peak_bytes,
                        pending_reads=len(self.pending), layers=len(self.layers),
                        transition_edges=len(self.transitions), budget_bytes=self.budget,
                        enabled=not self.closed and self.budget > 0,
                        strategy='observed_same_layer_transitions_then_last_route_reuse',
                        max_layers=self.max_layers, max_transition_edges=self.max_transitions,
                        max_pending_reads=self.max_pending, max_route_experts=self.max_route_experts,
                        byte_metric='decoded_payload_not_physical_ssd')

    def close(self):
        """Stop prediction admission and drain predictions; shared IO stays open."""
        with self.lock:
            self.closed = True
        self.discard()
        with self.lock:
            self.layers.clear()
            self.transitions.clear()
