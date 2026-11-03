"""Parser/admission fixtures only; no device, model or external PID is observed."""
from copy import deepcopy
import subprocess
import unittest

from research.glm53_gpu_v3.system_memory import (
    COUNTERS, PAGE_GAUGES, GIB, MIB, SystemMemoryObserver, compare_system,
    parse_sysctl, parse_vm_stat, partial_dense_cache_admission)


def vm_text():
    lines = ['Mach Virtual Memory Statistics: (page size of 16384 bytes)']
    lines += [f'{label}: 100.' for label in PAGE_GAUGES]
    lines += [f'"{label}": 200.' for label in COUNTERS]
    return '\n'.join(lines)


SYSCTL = ('hw.memsize: 17179869184\nvm.swapusage: total = 1024.00M used = 16.50M free = 1007.50M\n'
           'kern.memorystatus_vm_pressure_level: 1\nkern.boottime: { sec = 1000, usec = 50 } date\n')


def snapshot(start=0, swap=10 * MIB):
    return {'available': True, **parse_vm_stat(vm_text()), **parse_sysctl(SYSCTL),
            'monotonic_start': start, 'monotonic_end': start + .1, 'swap_used_bytes': swap}


class SystemMemoryTests(unittest.TestCase):
    def test_page_units_and_compressor_physical_logical_are_separate(self):
        result = parse_vm_stat(vm_text())
        self.assertEqual(result['gauges']['wired_bytes'], 100 * 16384)
        self.assertEqual(result['gauges']['compressor_physical_bytes'], 100 * 16384)
        self.assertEqual(result['gauges']['available_reclaimable_estimate_bytes'], 300 * 16384)
        self.assertEqual(result['cumulative_counters']['Pageins'], 200)
        self.assertIsNone(result['file_cache_hit_bytes'])
        with self.assertRaises(ValueError):
            parse_vm_stat(vm_text().replace('Swapouts', 'unrelated'))

    def test_sysctl_units_and_invalid_pressure_are_rejected(self):
        self.assertEqual(parse_sysctl(SYSCTL)['swap_used_bytes'], int(16.5 * MIB))
        self.assertEqual(parse_sysctl(SYSCTL)['boot_identity'], [1000, 50])
        with self.assertRaises(ValueError):
            parse_sysctl(SYSCTL.replace('level: 1', 'level: 0'))

    def test_snapshot_records_bracket_and_fails_closed_on_command_error(self):
        observer = SystemMemoryObserver(command=lambda argv: vm_text() if argv[0].endswith('vm_stat') else SYSCTL)
        result = observer.snapshot()
        self.assertTrue(result['available'])
        self.assertGreaterEqual(result['monotonic_end'], result['monotonic_start'])
        self.assertFalse(result['atomic_snapshot'])
        def failure(argv):
            raise subprocess.TimeoutExpired(argv, 5)
        self.assertFalse(SystemMemoryObserver(command=failure).snapshot()['available'])

    def test_global_delta_rejects_reboot_reset_and_overlapping_intervals(self):
        before, after = snapshot(), snapshot(6)
        after['cumulative_counters']['Pageins'] += 4
        self.assertEqual(compare_system(before, after)['cumulative_counter_delta']['Pageins'], 4)
        for mutation in ('reset', 'boot', 'time'):
            invalid = deepcopy(after)
            if mutation == 'reset': invalid['cumulative_counters']['Pageins'] = 1
            elif mutation == 'boot': invalid['boot_identity'] = [2000, 0]
            else: invalid['monotonic_start'] = 0
            self.assertFalse(compare_system(before, invalid)['available'])

    def test_partial_cache_admission_counts_source_duplication_and_existing_guards(self):
        samples = [snapshot(), snapshot(6)]
        for row in samples:
            row['gauges']['available_reclaimable_estimate_bytes'] = 5 * GIB
        result = partial_dense_cache_admission(samples, baseline_peak_rss_bytes=6 * GIB,
                                               cache_bytes=512 * MIB, inflight_bytes=64 * MIB)
        self.assertTrue(result['admitted'])
        self.assertEqual(result['conservative_increment_bytes'], 1088 * MIB)
        self.assertFalse(partial_dense_cache_admission(samples, baseline_peak_rss_bytes=8 * GIB,
                         cache_bytes=512 * MIB)['admitted'])
        with self.assertRaises(ValueError):
            partial_dense_cache_admission(samples, baseline_peak_rss_bytes=6 * GIB,
                                          cache_bytes=MIB, max_rss_bytes=9 * GIB)

    def test_admission_refuses_swap_spike_warning_or_too_short_observation(self):
        good = [snapshot(), snapshot(6)]
        for row in good:
            row['gauges']['available_reclaimable_estimate_bytes'] = 5 * GIB
        for reason in ('swap', 'pressure', 'time'):
            rows = deepcopy(good)
            if reason == 'swap': rows[-1]['swap_used_bytes'] += MIB
            elif reason == 'pressure': rows[-1]['pressure_level'] = 2
            else: rows[-1]['monotonic_start'] = 2
            self.assertFalse(partial_dense_cache_admission(rows, baseline_peak_rss_bytes=6 * GIB,
                             cache_bytes=256 * MIB)['admitted'])


if __name__ == '__main__':
    unittest.main()
