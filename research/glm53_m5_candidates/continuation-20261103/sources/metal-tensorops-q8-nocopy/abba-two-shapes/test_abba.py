"""Tiny metadata/array tests; no native CLI, model, compiler or GPU."""
from pathlib import Path
from array import array
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('abba_candidate',HERE/'abba.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class Abba(unittest.TestCase):
    def test_exact_two_shapes_order_and_unchanged_limits(self):
        p=a.plan()
        self.assertEqual([(r['M'],r['K'],r['N']) for r in p['cases']],[(8192,4096,8),(8192,4096,16)])
        self.assertEqual(p['order'],['CPU','GPU','GPU','CPU'])
        self.assertEqual((p['processes_max'],p['comparisons_per_case'],p['repetitions_per_completed_graph']),(8,4,5))
        self.assertEqual(p['child_rss_ceiling_bytes'],128<<20)
        self.assertEqual(p['controller_rss_ceiling_bytes'],128<<20)
        self.assertEqual((p['sampling_interval_seconds'],p['child_deadline_seconds'],p['global_deadline_seconds']),(.1,20,240))
        self.assertFalse(p['build_performed']);self.assertFalse(p['model_loaded'])

    def test_last_cpu_cannot_pass_using_metal_tolerance(self):
        row={'all_finite':True,'metal_tolerance_passed':True,'cpu_bitexact':False}
        self.assertTrue(a.comparison_passed(True,row))
        self.assertFalse(a.comparison_passed(False,row))
        row['all_finite']=False
        self.assertFalse(a.comparison_passed(True,row))

    def test_changed_actual_bytes_fail_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'synthetic-source';p.write_bytes(b'original fixture')
            pins={str(p):a.sha(p)};a.verify_file_pins(pins)
            p.write_bytes(b'changed fixture!')
            with self.assertRaisesRegex(ValueError,'input changed'):a.verify_file_pins(pins)

    def test_wrong_cached_gpu_contract_is_refused_before_import(self):
        with mock.patch.dict(sys.modules,{'gpu_contract':types.SimpleNamespace(__file__='/fixture/wrong/gpu_contract.py')}):
            with self.assertRaisesRegex(ValueError,'another candidate'):a.load_frozen()

    def test_original_raw_validator_checks_bits_finiteness_and_lengths(self):
        f=a.load_frozen()
        with tempfile.TemporaryDirectory() as directory:
            p,q=Path(directory)/'cpu.f32',Path(directory)/'gpu.f32'
            p.write_bytes(array('f',[1,2,3]).tobytes());q.write_bytes(p.read_bytes())
            result=f.compare_raw(p,q,3)
            self.assertTrue(a.comparison_passed(False,result))
            q.write_bytes(array('f',[1.00001,2,3]).tobytes())
            result=f.compare_raw(p,q,3)
            self.assertTrue(a.comparison_passed(True,result));self.assertFalse(a.comparison_passed(False,result))
            q.write_bytes(array('f',[float('nan'),2,3]).tobytes())
            with self.assertRaises(ValueError):f.compare_raw(p,q,3)
            q.write_bytes(b'bad')
            with self.assertRaises(ValueError):f.compare_raw(p,q,3)

    def test_default_main_never_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'plan.json'
            with mock.patch.object(sys,'argv',['abba.py','--plan-output',str(output)]),mock.patch.object(a,'execute',side_effect=AssertionError('execution forbidden')):
                self.assertEqual(a.main(),0)
            self.assertEqual(json.loads(output.read_text())['status'],'prepared_not_executed')


if __name__=='__main__':unittest.main()
