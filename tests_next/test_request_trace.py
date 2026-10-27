"""Stdlib-only trace checks, also runnable without Apple MLX or pytest fixtures."""
import importlib.util
import json
from pathlib import Path
import threading
import unittest

_PATH = Path(__file__).resolve().parents[1] / 'air_llm/airllm/nextgen/request_trace.py'
_SPEC = importlib.util.spec_from_file_location('isolated_request_trace', _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RequestTrace = _MODULE.RequestTrace


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


class RequestTraceTests(unittest.TestCase):
    def test_nested_spans_account_exclusive_and_union_without_double_counting(self):
        clock = Clock()
        trace = RequestTrace(clock=clock)
        with trace.span('request', category='request', phase='prefill'):
            clock.now = 10
            with trace.span('read', category='io'):
                clock.now = 30
                trace.count('logical_bytes', 50)
            clock.now = 40
            with trace.span('compute', category='submission'):
                clock.now = 70
            clock.now = 100
        result = trace.export()
        root = result['records'][0]
        self.assertEqual(root['inclusive_ns'], 100)
        self.assertEqual(root['exclusive_ns'], 50)
        self.assertEqual(result['summary']['observed_span_union_ns'], 100)
        self.assertEqual(result['summary']['phases']['prefill']['inclusive_sum_ns'], 150)
        self.assertEqual(result['summary']['phases']['prefill']['exclusive_sum_ns'], 100)
        self.assertEqual(result['summary']['phases']['prefill']['union_ns'], 100)
        self.assertEqual(result['counters']['logical_bytes'], 50)
        self.assertTrue(result['complete'])
        self.assertFalse(result['attribution']['pure_gpu_time'])
        json.dumps(result, allow_nan=False)

    def test_overlap_between_threads_is_counted_once_in_wall_union(self):
        times = {'MainThread': iter([0]), 'a': iter([10, 40]), 'b': iter([20, 50])}
        trace = RequestTrace(clock=lambda: next(times[threading.current_thread().name]))
        barrier = threading.Barrier(2)
        failures = []

        def worker():
            try:
                with trace.span('read', category='io', phase='prefill'):
                    barrier.wait(timeout=5)
            except BaseException as error:
                failures.append(error)

        threads = [threading.Thread(target=worker, name=name) for name in ['a', 'b']]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=6)
        self.assertEqual(failures, [])
        result = trace.export()
        self.assertEqual(result['summary']['categories']['io']['inclusive_sum_ns'], 60)
        self.assertEqual(result['summary']['categories']['io']['exclusive_sum_ns'], 60)
        self.assertEqual(result['summary']['categories']['io']['union_ns'], 40)
        self.assertEqual(result['summary']['observed_span_union_ns'], 40)
        self.assertEqual(len({record['thread_id'] for record in result['records']}), 2)

    def test_exception_closes_span_and_restores_parent_stack(self):
        clock = Clock()
        trace = RequestTrace(clock=clock)
        with trace.span('parent', phase='prefill'):
            with self.assertRaisesRegex(OSError, 'failure'):
                with trace.span('read', category='io'):
                    clock.now = 20
                    raise OSError('failure')
            trace.event('recovered')
            clock.now = 30
        with trace.span('decode', phase='decode'):
            clock.now = 40
        result = trace.export()
        self.assertTrue(result['complete'])
        self.assertEqual(result['records'][1]['outcome'], 'error')
        self.assertEqual(result['records'][1]['error_type'], 'OSError')
        self.assertEqual(result['records'][2]['parent_id'], result['records'][0]['id'])
        self.assertEqual(result['records'][2]['phase'], 'prefill')
        self.assertIsNone(result['records'][3]['parent_id'])

    def test_storage_bound_marks_partial_evidence_and_keeps_counters(self):
        clock = Clock()
        trace = RequestTrace(max_records=2, clock=clock)
        with trace.span('request'):
            with trace.span('kept'):
                clock.now = 5
            with trace.span('dropped'):
                clock.now = 10
                trace.event('dropped event')
                trace.count('reads', 1)
            clock.now = 20
        result = trace.export()
        self.assertEqual(len(result['records']), 2)
        self.assertEqual(result['dropped_records'], 2)
        self.assertFalse(result['complete'])
        self.assertEqual(result['counters']['reads'], 1)

    def test_live_export_is_incomplete_and_is_detached_from_future_updates(self):
        clock = Clock()
        trace = RequestTrace(clock=clock)
        with trace.span('request', phase='prefill', block=32):
            before = trace.export()
            clock.now = 10
        self.assertFalse(before['complete'])
        self.assertEqual(before['open_spans'], 1)
        self.assertIsNone(before['records'][0]['end_ns'])
        self.assertTrue(trace.export()['complete'])

    def test_disabled_trace_never_reads_clock_and_propagates_errors(self):
        def failing_clock():
            raise AssertionError('disabled clock accessed')
        trace = RequestTrace(enabled=False, clock=failing_clock)
        with self.assertRaises(ValueError):
            with trace.span('ignored'):
                trace.event('ignored')
                trace.count('ignored')
                raise ValueError('original')
        result = trace.export()
        self.assertEqual(result['records'], [])
        self.assertFalse(result['enabled'])
        self.assertFalse(result['complete'])

    def test_counter_rejects_nonfinite_values_and_invalid_bounds(self):
        trace = RequestTrace()
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                trace.count('x', value)
        with self.assertRaises(TypeError):
            trace.count('x', '1')
        for value in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                RequestTrace(max_records=value)


if __name__ == '__main__':
    unittest.main()
