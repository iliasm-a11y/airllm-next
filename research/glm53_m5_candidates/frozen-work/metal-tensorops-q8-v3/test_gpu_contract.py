"""Small metadata/trace/lifecycle fixtures only. No model, GPU or build."""
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import gpu_contract as gc


def profile():
    return gc.profiles.profile({'CPU_METAL_Q8_BRIDGE_COMPILED':True,
        'native_libraries_sha256':{'ggml-cpu':'1'*64,'ggml-base':'2'*64,'llama':'3'*64},
        'provider_patch_sha256':'4'*64,'kernel_sha256':'5'*64,
        'runner_source_sha256':'6'*64,'native_guard_policy_sha256':'7'*64})


def state():
    return {'execution_backend':gc.BACKEND,'gpu_execution_possible':True,'gpu_bridge_requested':True,
        'cpu_metal_q8_bridge_compiled':True,'gpu_policy_sha256':profile()['canonical_sha256'],
        'n_gpu_layers':0,'op_offload':False,'offload_kqv':False,'moe_cache_size':0,
        'permanent_gpu_weight_bytes':0,'weight_storage_policy':'cpu_originals_bounded_row_tiles',
        'n_ctx':2048,'n_seq_max':1,'n_batch':16,'n_ubatch':16,
        'prefill_chunk_policy':gc.TAIL_POLICY,'prefill_chunk_policy_version':1,'tail_reference_microbatch':8}


def trace(m=8192,k=4096,n=8):
    document=profile();lines=[
        f"AIRLLM_CPU_METAL_Q8 enabled=1 backend={gc.BACKEND} canonical_sha256={document['canonical_sha256']} kernel_sha256={'5'*64}",
        f"AIRLLM_CPU_METAL_Q8 allocation_count=4 gpu_payload_current={gc.GPU_BYTES} gpu_payload_peak={gc.GPU_BYTES} weight_tile_bytes={gc.WEIGHT_TILE_BYTES} native_swap_baseline_bytes=0 device_allocated_current={gc.GPU_BYTES} device_allocated_observed_peak={gc.GPU_BYTES} device_allocated_limit={gc.DATA_LIMIT}"]
    for index,first in enumerate(range(0,m,1536),1):
        count=min(1536,m-first)
        shape=f"weight=blk.3.attn_q_a.weight tile_first={first} tile_rows={count} M={m} K={k} N={n}"
        lines.append(f"AIRLLM_CPU_METAL_Q8 event=submitted actual_gpu=1 submission={index} {shape} pressure=2 available_bytes={2<<30} rss_bytes={1<<30} swap_growth_bytes=0")
        lines.append(f"AIRLLM_CPU_METAL_Q8 event=completed actual_gpu=1 submission={index} completed_commands={index} gpu_seconds=0.001 terminal_success=1")
        lines.append(f"AIRLLM_CPU_METAL_Q8 event=tile_done actual_gpu=1 attempt=1 {shape} copy_weight_bytes={count*(k//32)*34} copy_weight_us=3 gpu_seconds=0.001 command_and_readback_us=20 submissions={index} completed_commands={index} allocation_count=4 operation_buffer_reuses=0 tile_buffer_reuses={index-1} gpu_payload_current={gc.GPU_BYTES}")
    provider=((64+n*(k//32)*34+63)&~63)+m*n*4
    cpu=max(2<<20,provider)
    lines.append(f"AIRLLM_CPU_METAL_Q8 event=published publish=1 attempt=1 operations=1 submissions={index} completed_commands={index} actual_gpu=1 M={m} K={k} N={n} provider_workspace={provider} cpu_graph_workspace={cpu} gpu_workspace={gc.GPU_BYTES} total_workspace={cpu+gc.GPU_BYTES} device_allocated_current={gc.GPU_BYTES} device_allocated_observed_peak={gc.GPU_BYTES} device_allocated_limit={gc.DATA_LIMIT}")
    return '\n'.join(lines)+'\n'


class ContractTests(unittest.TestCase):
    def test_valid_trace_proves_six_completed_commands_and_one_publication(self):
        result=gc.parse_trace(trace(),profile(),baseline_swap=0)
        self.assertEqual(result['submitted_commands'],6)
        self.assertEqual(result['completed_commands'],6)
        self.assertEqual(result['published_operations'],1)

    def test_import_or_enable_without_command_is_rejected(self):
        with self.assertRaises(gc.GpuContractError) as caught:
            gc.parse_trace('\n'.join(trace().splitlines()[:2]),profile(),baseline_swap=0)
        self.assertEqual(caught.exception.code,'GPU_ZERO_COMMANDS')

    def test_modified_key_or_backend_is_rejected(self):
        for field,value in (('backend','native_cpu'),('n_gpu_layers',1)):
            document=profile();document['canonical_identity'][field]=value
            with self.assertRaises(gc.GpuContractError):gc.validate_profile(document)
        with self.assertRaises(gc.GpuContractError):gc.parse_trace(trace().replace('canonical_sha256=','canonical_sha256=0'),profile(),baseline_swap=0)

    def test_native_backend_and_compile_feature_must_agree(self):
        for patch in ({'execution_backend':'native_cpu'},{'cpu_metal_q8_bridge_compiled':False},
                {'gpu_policy_sha256':'0'*64},{'gpu_execution_possible':False}):
            with self.assertRaises(gc.GpuContractError):gc.validate_state({**state(),**patch},profile(),16)

    def test_potential_permanent_weight_copy_is_rejected(self):
        for patch in ({'n_gpu_layers':1},{'op_offload':True},{'offload_kqv':True},
                {'permanent_gpu_weight_bytes':1},{'weight_storage_policy':'MTL0_Mapped'},
                {'n_gpu_layers':False}):
            with self.assertRaises(gc.GpuContractError) as caught:gc.validate_state({**state(),**patch},profile(),16)
            self.assertEqual(caught.exception.code,'GPU_WEIGHT_PLACEMENT')

    def test_legacy_16_tail_driver_is_rejected(self):
        for patch in ({'prefill_chunk_policy':'fixed-microbatch-v0'},{'tail_reference_microbatch':16}):
            with self.assertRaises(gc.GpuContractError) as caught:gc.validate_state({**state(),**patch},profile(),16)
            self.assertEqual(caught.exception.code,'GPU_TAIL_POLICY')

    def test_inflight_or_missing_publication_is_rejected(self):
        for value in (trace().replace('event=completed','event=unobserved'),
                '\n'.join(trace().splitlines()[:-1]),trace().replace('completed_commands=6','completed_commands=5')):
            with self.assertRaises(gc.GpuContractError):gc.parse_trace(value,profile(),baseline_swap=0)

    def test_unbounded_copy_reuse_or_workspace_is_rejected(self):
        for value in (trace().replace('copy_weight_bytes=6684672','copy_weight_bytes=99000000'),
                trace().replace('tile_buffer_reuses=5','tile_buffer_reuses=0'),
                trace().replace(f'total_workspace={2*1024**2+gc.GPU_BYTES}','total_workspace=99000000'),
                trace().replace('gpu_payload_peak=8028160','gpu_payload_peak=99000000')):
            with self.assertRaises(gc.GpuContractError):gc.parse_trace(value,profile(),baseline_swap=0)

    def test_nonfinite_timing_is_not_evidence(self):
        with self.assertRaises(gc.GpuContractError):gc.parse_trace(trace().replace('gpu_seconds=0.001','gpu_seconds=nan'),profile(),baseline_swap=0)

    def test_actual_loaded_image_path_must_match_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            libraries=Path(directory);fingerprints=profile()['canonical_identity']['fingerprints']
            for name in ('ggml-cpu','ggml-base','llama'):
                path=libraries/('lib'+name+'.dylib');path.write_bytes(name.encode())
                fingerprints['native_libraries_sha256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
            document=gc.profiles.profile(fingerprints)
            observed={'loaded_native_images':[str(libraries/('lib'+name+'.dylib')) for name in ('ggml-cpu','ggml-base','llama')]}
            self.assertTrue(gc.validate_loaded_images(observed,document,libraries)['passed'])
            wrong=deepcopy(observed);wrong['loaded_native_images'][0]='/outside/libggml-cpu.dylib'
            with self.assertRaises(gc.GpuContractError):gc.validate_loaded_images(wrong,document,libraries)

    def test_sufficient_pressure_two_and_exact_boundary_are_admitted(self):
        floor=gc.profiles.NATIVE_MEMORY_POLICY['reserve_bytes']+gc.ADDITIONAL_RESERVATION+gc.profiles.NATIVE_MEMORY_POLICY['warning_headroom_bytes']
        sample={'pressure_level':2,'available_bytes':floor,'rss_bytes':int(8.5*(1<<30)),'swap_growth_bytes':32<<20}
        self.assertTrue(gc.sufficient_memory_sample(sample,0))
        gc.parse_trace(trace(),profile(),baseline_swap=0)
        for patch in ({'available_bytes':floor-1},{'rss_bytes':sample['rss_bytes']+1},{'swap_growth_bytes':(32<<20)+1},
                {'pressure_level':4},{'pressure_level':3},{'available_bytes':float('nan')},{'rss_bytes':None},{'pressure_level':True}):
            self.assertFalse(gc.sufficient_memory_sample({**sample,**patch},0))

    def test_warning_headroom_missing_sample_or_wrong_swap_baseline_is_rejected(self):
        for value in (trace().replace(f'available_bytes={2<<30}',f'available_bytes={1024<<20}'),
                trace().replace('pressure=2','pressure=4'),trace().replace('rss_bytes=1073741824','rss_bytes=nan'),
                trace().replace('swap_growth_bytes=0','swap_growth_bytes=33554433'),
                trace().replace('native_swap_baseline_bytes=0','native_swap_baseline_bytes=1')):
            with self.assertRaises(gc.GpuContractError):gc.parse_trace(value,profile(),baseline_swap=0)

    def test_limits_and_dynamic_context_are_exact_and_not_math_key_inputs(self):
        limits=dict(gc.profiles.NATIVE_MEMORY_POLICY)
        gc.validate_limits(limits,profile())
        for patch in ({'max_swap_growth_bytes':64<<20},{'reserve_bytes':0},{'warning_headroom_bytes':float('nan')},
                {'warning_grace_seconds':21}):
            with self.assertRaises(gc.GpuContractError):gc.validate_limits({**limits,**patch},profile())
        for value in (None,{}, {'schema':1,'gpu_bridge_requested':True,'native_guard_swap_baseline_bytes':1}):
            with self.assertRaises(gc.GpuContractError):gc.validate_resource_context(value,0)
        gc.validate_resource_context({'schema':1,'gpu_bridge_requested':True,'native_guard_swap_baseline_bytes':0},0)
        self.assertNotIn('native_guard_swap_baseline_bytes',profile()['canonical_identity'])

    def test_device_measurement_is_separate_bounded_and_required(self):
        for value in (trace().replace(f'device_allocated_current={gc.GPU_BYTES}','device_allocated_current=0'),
                trace().replace(f'device_allocated_observed_peak={gc.GPU_BYTES}','device_allocated_observed_peak=99000000'),
                trace().replace(f'device_allocated_limit={gc.DATA_LIMIT}','device_allocated_limit=99000000')):
            with self.assertRaises(gc.GpuContractError):gc.parse_trace(value,profile(),baseline_swap=0)
        result=gc.parse_trace(trace(),profile(),baseline_swap=0)
        self.assertEqual(result['device_allocated_observed_peak_bytes'],gc.GPU_BYTES)
        self.assertIsNone(result['unmeasured_driver_residency_bytes'])

    def test_capture_params_must_match_manifest_without_dynamic_envelopes(self):
        report={'state_parameters':state(),'native_linkage':{},'native_resource_context':{
            'schema':1,'gpu_bridge_requested':True,'native_guard_swap_baseline_bytes':0}}
        metadata={'snapshot_manifest':{'state_parameters':state()}}
        with mock.patch.object(gc,'validate_loaded_images',return_value={'passed':True}):
            self.assertTrue(gc.validate_capture_report(report,metadata,profile(),'/fixture/lib',0,16)['passed'])
            for patch in ({'state_parameters':{**state(),'native_linkage':{}}},
                    {'state_parameters':{**state(),'n_ubatch':8}}, {'native_resource_context':None}):
                with self.assertRaises(gc.GpuContractError):
                    gc.validate_capture_report({**report,**patch},metadata,profile(),'/fixture/lib',0,16)


    def test_build_provenance_requires_all_roles_and_before_after_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.cpp';path.write_text('original')
            hashes={str(path):gc.sha(path)}
            receipt={'schema_version':2,'input_roles':{'source':str(path)},'inputs_sha256':hashes,
                'inputs_before':dict(hashes),'inputs_after':dict(hashes)}
            gc.validate_input_provenance(receipt,('source',))
            for value in ({**receipt,'inputs_after':{}},{**receipt,'schema_version':1},
                    {**receipt,'input_roles':{}},{**receipt,'inputs_before':{}}):
                with self.assertRaises(gc.GpuContractError):gc.validate_input_provenance(value,('source',))
            path.write_text('changed')
            with self.assertRaises(gc.GpuContractError):gc.validate_input_provenance(receipt,('source',))



class Gate:
    def __init__(self,*unused):pass
    def sample(self,**unused):return {'passed':True}
    def summary(self):return {'passed':True}

class Battery(Gate):
    def sample(self,**unused):return {'passed':False}

class Child:
    pid=71001
    returncode=None
    def poll(self):return self.returncode
    def send_signal(self,value):self.returncode=-value
    def wait(self,**unused):return self.returncode

class Awake:
    def __init__(self,*unused):self.record={}
    def __enter__(self):return self
    def check(self):pass
    def close(self):pass


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('isolated_native_gpu_run',HERE/'native_gpu_run.py')
        cls.runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.runner)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.document=self.root/'profile.json';self.document.write_text(json.dumps(profile()))
        (self.root/'provider-build').write_text('{}')

    def tearDown(self):self.temp.cleanup()

    def config(self):
        n=self.runner
        return {'model_path':'/fixture/model','runtime_binary':'/fixture/server','runtime_manifest_file':'/fixture/manifest',
            'verification_file':'/fixture/receipt','context_tokens':2048,'batch_size':16,'threads':10,'gpu_layers':0,
            'limits':{**vars(n.NativeLimits()),**gc.profiles.NATIVE_MEMORY_POLICY}}

    def call(self,**kwargs):
        return self.runner.run(self.root/'config','/fixture/helper','/fixture/helper.cpp','/fixture/prompts','/fixture/lib',self.root/'out',
            gpu_profile=self.document,provider_build=self.root/'provider-build',driver_build=self.root/'driver-build',microbatch=16,**kwargs)

    def test_battery_never_prepares_or_acquires_uid_or_starts_native(self):
        n=self.runner
        with mock.patch.object(n.qa,'configuration',return_value=(self.config(),'a'*64)), \
             mock.patch.object(n.qa,'BENCHMARK_LOCK',self.root/'lock'),mock.patch.object(n,'prepare') as prepare, \
             mock.patch.object(n,'acquire_uid_model_lock') as uid:
            report=self.call(gate_factory=Battery)
        self.assertEqual(report['error_code'],'POWER_CONTEXT_UNAVAILABLE');prepare.assert_not_called();uid.assert_not_called()

    def test_engine_busy_retains_class_without_model_spawn(self):
        n=self.runner
        with mock.patch.object(n.qa,'configuration',return_value=(self.config(),'a'*64)), \
             mock.patch.object(n.qa,'BENCHMARK_LOCK',self.root/'lock'),mock.patch.object(n,'prepare') as prepare, \
             mock.patch.object(n,'acquire_uid_model_lock',side_effect=n.qa.QualificationError('ENGINE_BUSY','fixture')):
            report=self.call(gate_factory=Gate)
        self.assertEqual(report['error_code'],'ENGINE_BUSY');prepare.assert_not_called()

    def simulate(self,returncode,log,*,guard_code=None,deadline=False,observed_fd=True):
        n=self.runner;child=Child();seen={}
        holder=mock.Mock();holder.fileno.return_value=22
        uid={'pass_fd':22,'device':1,'inode':2}
        baseline={'available_bytes':4*n.GiB,'swap_used_bytes':0,'pressure_level':1}
        def sample(pid=None):
            if pid is not None:child.returncode=returncode
            return {**baseline,**({'rss_bytes':n.GiB} if pid else {})}
        def popen(*args,**kwargs):
            seen.update(kwargs);kwargs['stderr'].write(log.encode());kwargs['stderr'].flush()
            capture=self.root/'out/capture';capture.mkdir()
            (capture/'report.json').write_text(json.dumps({'state_parameters':state(),'native_linkage':{},
                'native_resource_context':{'schema':1,'gpu_bridge_requested':True,'native_guard_swap_baseline_bytes':0},'requests':[]}))
            return child
        target=SimpleNamespace(create_time=lambda:123,exe=lambda:'/fixture/helper',is_running=lambda:True)
        metadata={'snapshot_manifest':{'state_parameters':state()},'tracked_identities':{}}
        with mock.patch.object(n.qa,'configuration',return_value=(self.config(),'a'*64)), \
             mock.patch.object(n.qa,'BENCHMARK_LOCK',self.root/'lock'),mock.patch.object(n,'prepare',return_value=(metadata,{'agents':[]})), \
             mock.patch.object(n,'acquire_uid_model_lock',return_value=(holder,uid)), \
             mock.patch.object(n,'observe_uid_fd',return_value={'observed_matching_lock_fd':observed_fd}), \
             mock.patch.object(n.gc,'read_json',side_effect=lambda p,*unused:json.loads(Path(p).read_text()) if Path(p).is_file() else {}), \
             mock.patch.object(n.gc,'validate_artifacts',return_value={'passed':True}), \
             mock.patch.object(n.gc,'validate_loaded_images',return_value={'passed':True}), \
             mock.patch.object(n,'resource_snapshot',side_effect=sample), \
             mock.patch.object(n,'ProcessIOObserver',return_value=SimpleNamespace(snapshot=lambda:{'available':False})), \
             mock.patch.object(n,'power_evidence_matches',return_value=True),mock.patch.object(n.time,'sleep'):
            with mock.patch.object(n.NativeGlmModel,'_guard_resource_sample',
                    side_effect=lambda guard,sample,*unused:guard_code if sample.get('rss_bytes') else None):
                if deadline:
                    with mock.patch.object(n.time,'perf_counter',side_effect=itertools.count(0,2).__next__):
                        report=self.call(gate_factory=Gate,awake_factory=Awake,popen_factory=popen,process_factory=lambda pid:target,max_seconds=1)
                else:
                    report=self.call(gate_factory=Gate,awake_factory=Awake,popen_factory=popen,process_factory=lambda pid:target)
        self.assertEqual(seen['pass_fds'],(22,));self.assertTrue(report['uid_model_lock']['parent_fd_closed']);holder.close.assert_called_once()
        return report

    def test_zero_native_exit_without_gpu_commands_fails(self):
        report=self.simulate(0,'\n'.join(trace().splitlines()[:2]))
        self.assertEqual(report['status'],'failed');self.assertEqual(report['returncode'],0)
        self.assertEqual(report['error_code'],'GPU_ZERO_COMMANDS');self.assertFalse(report['actual_gpu'])

    def test_nonzero_exit_not_replaced_by_secondary_trace_failure(self):
        report=self.simulate(-6,'fatal fixture\n')
        self.assertEqual(report['status'],'failed');self.assertEqual(report['returncode'],-6)
        self.assertEqual(report['error_code'],'NATIVE_NONZERO_EXIT');self.assertIn('gpu_trace_error',report)

    def test_fixture_completed_trace_is_recorded_without_global_qualification(self):
        report=self.simulate(0,trace())
        self.assertEqual(report['status'],'completed');self.assertTrue(report['actual_gpu'])
        self.assertEqual(report['gpu_trace']['completed_commands'],6)
        self.assertFalse(report['snapshot_qualified']);self.assertFalse(report['functional_agents_qualified'])

    def test_resource_failure_class_is_not_hidden_by_trace_failure(self):
        report=self.simulate(None,'',guard_code='RSS_LIMIT')
        self.assertEqual(report['error_code'],'RSS_LIMIT');self.assertEqual(report['returncode'],-signal.SIGTERM)

    def test_deadline_class_is_not_hidden_by_trace_failure(self):
        report=self.simulate(None,'',deadline=True)
        self.assertEqual(report['error_code'],'DEADLINE');self.assertEqual(report['returncode'],-signal.SIGTERM)

    def test_uid_descriptor_observation_is_required(self):
        report=self.simulate(None,'',observed_fd=False)
        self.assertEqual(report['error_code'],'UID_LOCK_INHERITANCE');self.assertEqual(report['returncode'],-signal.SIGTERM)

    def test_gpu_additional_budget_cannot_be_omitted(self):
        n=self.runner;c=self.config()
        admitted=n.admit(c,16,'full',False,{'available_bytes':4*n.GiB,'pressure_level':1})
        self.assertEqual(admitted['gpu_additional_reservation_bytes'],128*n.MiB)
        for value in (None,float('nan'),float('inf'),True):
            with self.assertRaises(n.qa.QualificationError):n.admit(c,16,'full',False,{'available_bytes':value,'pressure_level':1})


if __name__=='__main__':unittest.main()
