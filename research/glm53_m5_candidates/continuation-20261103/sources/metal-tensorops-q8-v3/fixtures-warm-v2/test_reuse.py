"""No hardware: reject missing iterations, nonfinite timings and changed bits."""
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('warm_fixture',HERE/'provider_fixture.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

class ReuseTests(unittest.TestCase):
    def example(self):
        return {'repetitions':5,'iteration_seconds':[1.,.2,.4,.1,.3],
                'iteration_sha256':['a'*64]*5,'output_sha256':'a'*64}
    def test_reports_cold_and_resident_median_separately(self):
        result=f.validate_repetitions(self.example())
        self.assertEqual(result['first_seconds'],1.)
        self.assertAlmostEqual(result['warm_median_seconds'],.25)
    def test_changed_output_in_any_repetition_refuses(self):
        for index in range(5):
            row=self.example();row['iteration_sha256'][index]='b'*64
            with self.assertRaises(f.gc.GpuContractError):f.validate_repetitions(row)
    def test_missing_nonfinite_and_negative_timing_refuse(self):
        for times in ([1.], [1.,2.,3.,4.,float('nan')], [1.,2.,3.,4.,float('inf')], [1.,2.,3.,4.,-1.]):
            row=self.example();row['iteration_seconds']=times
            with self.assertRaises(f.gc.GpuContractError):f.validate_repetitions(row)
    def test_cap_and_excluded_shapes_are_explicit(self):
        plan=f.plan()
        self.assertEqual(len(plan['cases']),21)
        self.assertEqual(plan['child_rss_ceiling_bytes'],128<<20)
        self.assertTrue(all(not(c['M']==8192 and c['K']==4096 and c['N']>=64) for c in plan['cases']))
        self.assertTrue(any(c['expectation']=='fatal_after_terminal_tile' for c in plan['cases']))

if __name__=='__main__':unittest.main()
