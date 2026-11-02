"""Mocked spawn/setup/signal checks; no native process, model or GPU executes."""
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import signal
import subprocess
import unittest

from research.glm53_gpu_v2.benchmark_lifecycle import account_final_swap, owned_server, stop_owned_server, stop_for_guard, interrupt_on_sigterm


class BenchmarkLifecycleTests(unittest.TestCase):
    def test_setup_failure_and_interrupt_both_clean_owned_child_and_record_failure(self):
        for error in (RuntimeError('metadata setup failed'), KeyboardInterrupt()):
            child = Mock(pid=1234)
            spawn, stop, finalized = Mock(return_value=child), Mock(), Mock()
            with self.assertRaises(type(error)):
                with owned_server(['fixture-server'], stdout=None, env={}, popen=spawn,
                                  stop_server=stop, after_cleanup=finalized):
                    raise error
            stop.assert_called_once_with(child)
            self.assertIs(finalized.call_args.args[0], child)
            self.assertIs(finalized.call_args.args[1], error)
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])

    def test_failed_spawn_records_failure_without_signalling_unowned_process(self):
        error = OSError('fixture spawn failure')
        stop, finalized = Mock(), Mock()
        with self.assertRaises(OSError):
            with owned_server(['fixture'], stdout=None, env={}, popen=Mock(side_effect=error),
                              stop_server=stop, after_cleanup=finalized):
                self.fail('failed spawn cannot enter the setup body')
        stop.assert_not_called()
        finalized.assert_called_once_with(None, error)

    def test_owned_group_term_and_bounded_kill_fallback(self):
        child = Mock(pid=4321)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('fixture', 10), 0]
        kill = Mock()
        stop_owned_server(child, kill_group=kill)
        self.assertEqual(kill.call_args_list[0].args, (4321, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (4321, signal.SIGKILL))
        self.assertTrue(all(call.kwargs == {'timeout': 10} for call in child.wait.call_args_list))
        child = Mock()
        child.poll.return_value = 0
        kill.reset_mock()
        stop_owned_server(child, kill_group=kill)
        kill.assert_not_called()

    def test_final_swap_growth_is_included_after_watcher_stops(self):
        report = {'swap_start_bytes': 100, 'peak_swap_growth_bytes': 10}
        account_final_swap(report, 146)
        self.assertEqual(report['swap_final_bytes'], 146)
        self.assertEqual(report['peak_swap_growth_bytes'], 46)
        account_final_swap(report, 80)
        self.assertEqual(report['peak_swap_growth_bytes'], 46)
        clean = {'swap_start_bytes': 100, 'peak_swap_growth_bytes': 0}
        account_final_swap(clean, 90)
        self.assertEqual(clean['peak_swap_growth_bytes'], 0)

    def test_guard_breach_and_monitor_error_escalate_stubborn_child_before_reader_cleanup(self):
        for violation in ({'swap_used': 1000}, {'monitor_error': 'fixture pressure query failed'}):
            child = Mock(pid=1234)
            child.poll.return_value = None
            child.wait.side_effect = [subprocess.TimeoutExpired('fixture', 10), 0]
            kill = Mock()
            report = {'guard_violation': None}
            stop_for_guard(child, report, violation,
                           stop_server=lambda p: stop_owned_server(p, kill_group=kill))
            self.assertIs(report['guard_violation'], violation)
            self.assertEqual([call.args for call in kill.call_args_list],
                             [(1234, signal.SIGTERM), (1234, signal.SIGKILL)])
            self.assertEqual([call.kwargs for call in child.wait.call_args_list],
                             [{'timeout': 10}, {'timeout': 10}])
            self.assertNotIn('guard_stop_error', report)
        report = {'guard_violation': violation}
        stop_for_guard(child, report, {'monitor_error': 'secondary'},
                       stop_server=Mock(side_effect=RuntimeError('cleanup failed')))
        self.assertIs(report['guard_violation'], violation)
        self.assertIn('cleanup failed', report['guard_stop_error'])

    def test_monitor_and_finally_cleanup_are_serialized_for_the_same_child(self):
        class Child:
            pid = 4321
            exited = False
            entered = Event()
            release = Event()

            def poll(self):
                return 0 if self.exited else None

            def wait(self, timeout):
                self.entered.set()
                if not self.release.wait(2):
                    raise AssertionError('fixture did not release owned wait')
                self.exited = True
                return 0

        child = Child()
        kill = Mock()
        with ThreadPoolExecutor(2) as pool:
            monitor = pool.submit(stop_owned_server, child, kill_group=kill)
            self.assertTrue(child.entered.wait(2))
            cleanup = pool.submit(stop_owned_server, child, kill_group=kill)
            child.release.set()
            monitor.result(timeout=2)
            cleanup.result(timeout=2)
        kill.assert_called_once_with(4321, signal.SIGTERM)

    def test_standalone_sigterm_runs_owned_cleanup_and_restores_previous_handler(self):
        child = Mock(pid=1234)
        stop, finalized = Mock(), Mock()
        previous = object()
        with patch('research.glm53_gpu_v2.benchmark_lifecycle.signal.signal', return_value=previous) as install:
            with self.assertRaises(KeyboardInterrupt):
                with interrupt_on_sigterm():
                    handler = install.call_args.args[1]
                    with owned_server(['fixture'], stdout=None, env={}, popen=Mock(return_value=child),
                                      stop_server=stop, after_cleanup=finalized):
                        handler(signal.SIGTERM, None)
            self.assertEqual(install.call_count, 2)
            self.assertEqual(install.call_args.args, (signal.SIGTERM, previous))
        stop.assert_called_once_with(child)
        self.assertIsInstance(finalized.call_args.args[1], KeyboardInterrupt)


if __name__ == '__main__':
    unittest.main()
