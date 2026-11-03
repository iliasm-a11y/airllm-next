"""libproc ABI, identity and reset fixtures; no live external process queries."""
from copy import deepcopy
import ctypes
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from research.glm53_gpu_v3.process_io import ProcessIOObserver, TaskInfo, TaskVMRev1, UsageV4, compare_process, sample_owned


def fake_libraries(*, start=100, support_v4=True, fail=False):
    def rusage(pid, flavor, pointer):
        if fail or flavor == 4 and not support_v4:
            return -1
        data = pointer._obj
        for name, _ in data._fields_:
            if name != 'uuid': setattr(data, name, 10)
        data.start_abstime = start
        return 0
    def task(pid, flavor, arg, pointer, size):
        for name, _ in pointer._obj._fields_: setattr(pointer._obj, name, 10)
        return size
    def timebase(pointer):
        pointer._obj.numer, pointer._obj.denom = 125, 3
        return 0
    return SimpleNamespace(proc_pid_rusage=Mock(side_effect=rusage), proc_pidinfo=Mock(side_effect=task)), SimpleNamespace(mach_timebase_info=Mock(side_effect=timebase))


def snapshot(start=0):
    return {'available': True, 'identity': {'pid': 1234, 'start_abstime': 100},
            'monotonic_start': start, 'monotonic_end': start + .1,
            'rusage': {k: 10 for k in ('user_ticks', 'system_ticks', 'pageins', 'os_storage_read_bytes',
                'os_storage_write_bytes', 'resident_bytes', 'physical_footprint_bytes', 'wired_bytes')},
            'task_info': {'available': True, **{k: 10 for k in ('faults', 'pageins', 'cow_faults', 'context_switches')}},
            'cpu_seconds': 1.}


class ProcessIOTests(unittest.TestCase):
    def test_ctypes_layouts_match_inspected_sdk_prefixes(self):
        self.assertEqual(ctypes.sizeof(UsageV4), 296)
        self.assertEqual(UsageV4.os_storage_read_bytes.offset, 144)
        self.assertEqual(ctypes.sizeof(TaskInfo), 96)
        self.assertEqual(TaskInfo.faults.offset, 52)
        self.assertEqual(ctypes.sizeof(TaskVMRev1), 152)
        self.assertEqual(TaskVMRev1.compressed_bytes.offset, 120)

    def test_snapshot_v4_and_v2_fallback_keep_nand_and_external_compression_unknown(self):
        for v4 in (True, False):
            proc, system = fake_libraries(support_v4=v4)
            result = ProcessIOObserver(1234, libproc=proc, libsystem=system).snapshot()
            self.assertTrue(result['available'])
            self.assertEqual(result['rusage']['flavor'], 4 if v4 else 2)
            self.assertEqual(result['identity'], {'pid': 1234, 'start_abstime': 100})
            self.assertIsNone(result['physical_nand_read_bytes'])
            self.assertIsNone(result['task_vm']['compressed_bytes'])
            self.assertEqual(result['mach_timebase'], {'numer': 125, 'denom': 3})

    def test_unavailable_or_reused_pid_fails_closed(self):
        proc, system = fake_libraries(fail=True)
        self.assertFalse(ProcessIOObserver(1234, libproc=proc, libsystem=system).snapshot()['available'])
        proc, system = fake_libraries()
        observer = ProcessIOObserver(1234, libproc=proc, libsystem=system)
        self.assertTrue(observer.snapshot()['available'])
        proc.proc_pid_rusage.side_effect = fake_libraries(start=200)[0].proc_pid_rusage.side_effect
        self.assertFalse(observer.snapshot()['available'])

    def test_pid_identity_race_inside_snapshot_and_partial_task_info_are_explicit(self):
        proc, system = fake_libraries()
        original = proc.proc_pid_rusage.side_effect
        calls = [0]
        def race(pid, flavor, pointer):
            result = original(pid, flavor, pointer)
            calls[0] += 1
            pointer._obj.start_abstime = calls[0] * 100
            return result
        proc.proc_pid_rusage.side_effect = race
        result = ProcessIOObserver(1234, libproc=proc, libsystem=system).snapshot()
        self.assertFalse(result['available'])
        self.assertIn('inside the snapshot', result['error'])
        proc, system = fake_libraries()
        proc.proc_pidinfo.side_effect = lambda *args: 0
        result = ProcessIOObserver(1234, libproc=proc, libsystem=system).snapshot()
        self.assertTrue(result['available'])
        self.assertFalse(result['task_info']['available'])

    def test_delta_keeps_os_reads_and_fault_counts_without_inventing_wait(self):
        before, after = snapshot(), snapshot(2)
        after['rusage']['os_storage_read_bytes'] += 4096
        after['task_info']['faults'] += 2
        after['rusage']['resident_bytes'] -= 5
        result = compare_process(before, after)
        self.assertTrue(result['available'])
        self.assertEqual(result['rusage_counter_delta']['os_storage_read_bytes'], 4096)
        self.assertEqual(result['task_counter_delta']['faults'], 2)
        self.assertEqual(result['rss_change_bytes'], -5)
        self.assertNotIn('weight_wait_seconds', result)

    def test_delta_rejects_pid_reuse_reset_wrap_or_overlapping_samples(self):
        before, good = snapshot(), snapshot(2)
        for reason in ('pid', 'io', 'fault', 'time'):
            after = deepcopy(good)
            if reason == 'pid': after['identity']['start_abstime'] += 1
            elif reason == 'io': after['rusage']['os_storage_read_bytes'] -= 1
            elif reason == 'fault': after['task_info']['faults'] -= 1
            else: after['monotonic_start'] = 0
            self.assertFalse(compare_process(before, after)['available'])

    def test_composite_sample_is_bracketed_without_process_suspension(self):
        process, system = Mock(), Mock()
        process.snapshot.return_value = {'available': True, 'pid': 1234}
        system.snapshot.return_value = {'available': True}
        result = sample_owned(process, system)
        self.assertTrue(result['available'])
        self.assertGreaterEqual(result['monotonic_end'], result['monotonic_start'])
        self.assertIn('non-atomic', result['scope'])
        self.assertIsNone(result['global_block_device'])
        counter = Mock(return_value={'available': True, 'scope': 'global device only'})
        observed = sample_owned(process, system, block_device_observer=counter)
        self.assertEqual(observed['global_block_device']['scope'], 'global device only')
        counter.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
