"""Portable metadata/raw fixtures; never imports OS runner or executes native."""
from array import array
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('provider_fixtures',HERE/'provider_fixture.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

class FixtureTests(unittest.TestCase):
    def test_suite_is_bounded_and_covers_shapes_tails_and_errors(self):
        plan=f.plan();self.assertEqual(len(plan['cases']),23);self.assertEqual(plan['processes_max'],46)
        self.assertLessEqual(plan['max_logical_fixture_payload_bytes'],64<<20)
        for k in (2048,4096):
            self.assertTrue(all(any(c['M']==8192 and c['K']==k and c['N']==n for c in plan['cases']) for n in (8,16,32,64,128)))
            self.assertTrue(all(any(c['K']==k and c['N']==n and c['expectation']=='cpu_fallback' for c in plan['cases']) for n in (1,6,14)))
        self.assertIsNone(plan['measured_elapsed_seconds']);self.assertFalse(plan['use_ref']);self.assertFalse(plan['full_model_qualified'])
        self.assertTrue(any(c['expectation']=='fatal_after_terminal_tile' for c in plan['cases']))

    def compare(self,a,b):
        with tempfile.TemporaryDirectory() as directory:
            aa=Path(directory)/'a';bb=Path(directory)/'b';aa.write_bytes(array('f',a).tobytes());bb.write_bytes(array('f',b).tobytes())
            return f.compare_raw(aa,bb,len(a))

    def test_float_bits_and_metal_tolerance_remain_distinct(self):
        row=self.compare([0.0,1.0],[-0.0,1.00001]);self.assertFalse(row['cpu_bitexact']);self.assertTrue(row['metal_tolerance_passed'])
        self.assertEqual(row['different_float32_bits_count'],2)

    def test_large_error_and_nonfinite_are_not_hidden(self):
        self.assertFalse(self.compare([1.0],[2.0])['metal_tolerance_passed'])
        for value in (float('nan'),float('inf')):
            with self.assertRaises(f.gc.GpuContractError):self.compare([1.0],[value])

    def test_raw_lengths_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a';b=Path(directory)/'b';a.write_bytes(b'\0'*4);b.write_bytes(b'\0'*3)
            with self.assertRaises(f.gc.GpuContractError):f.compare_raw(a,b,1)

    def test_zero_gpu_is_required_for_cpu_cases(self):
        fingerprints={'CPU_METAL_Q8_BRIDGE_COMPILED':True,'native_libraries_sha256':{'ggml-cpu':'1'*64,'ggml-base':'2'*64,'llama':'3'*64},
            'provider_patch_sha256':'4'*64,'kernel_sha256':'5'*64,'runner_source_sha256':'6'*64,'native_guard_policy_sha256':'7'*64}
        profile=f.gc.profiles.profile(fingerprints)
        enabled=f"AIRLLM_CPU_METAL_Q8 enabled=1 canonical_sha256={profile['canonical_sha256']} kernel_sha256={'5'*64}\n"
        self.assertFalse(f.zero_trace(enabled,profile,0,False)['actual_gpu'])
        with self.assertRaises(f.gc.GpuContractError):f.zero_trace(enabled+'AIRLLM_CPU_METAL_Q8 event=submitted actual_gpu=1\n',profile,0,False)

    def test_expected_fatal_requires_complete_bound_terminal_trace_prefix(self):
        import test_gpu_contract as prior
        profile=prior.profile();case={'M':2048,'K':4096,'N':8}
        prefix='\n'.join(prior.trace(m=2048,k=4096,n=8).splitlines()[:5]).replace('blk.3.attn_q_a.weight','fixture.q8.weight')
        result=f.fatal_trace(prefix,profile,0,case)
        self.assertEqual(result['completed_commands'],1);self.assertEqual(result['published_operations'],0)
        for value in (prefix.replace('canonical_sha256=','canonical_sha256=0'),
                prefix.replace('gpu_payload_peak=8028160','gpu_payload_peak=99000000'),
                prefix.replace('device_allocated_observed_peak=8028160','device_allocated_observed_peak=99000000'),
                prefix.replace('fixture.q8.weight','another.weight'),prefix.replace('tile_first=0','tile_first=8'),
                prefix.replace('event=completed','event=missing'),prefix.replace('tile_rows=1536','tile_rows=512'),
                prefix.replace('pressure=2','pressure=4'),prefix.replace('gpu_seconds=0.001','gpu_seconds=nan')):
            with self.assertRaises(f.gc.GpuContractError):f.fatal_trace(value,profile,0,case)


    def test_cpp_connects_actual_mapped_weights_normal_backend_and_publication_observer(self):
        source=(HERE/'provider_fixture.cpp').read_text()
        for needle in ('ggml_backend_cpu_buffer_from_ptr(weights,weight_bytes)','GGML_BACKEND_BUFFER_USAGE_WEIGHTS',
                'ggml_backend_cpu_set_n_threads(backend,THREADS)','ggml_backend_cpu_set_use_ref(backend,false)',
                'ggml_backend_graph_compute(backend,graph)','ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float',
                'ggml_set_abort_callback(fatal_observer)','fatal_dst[i]==0xa5','std::_Exit(86)',"token!='G'"):
            self.assertIn(needle,source)
        self.assertNotIn('llama_model_load',source)
        for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))

if __name__=='__main__':unittest.main()
