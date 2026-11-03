"""Prepared portable negatives; NOT executed during the active CPU32 model."""
import copy
import unittest
from unittest.mock import patch
import gpu_contract as g


def registry():
    return {'policy':g.CPU_ACCELERATOR_POLICY,'registry_before':['MTL','BLAS','CPU'],
        'registry_after':['MTL','CPU'],'BLAS_registered_after':False,
        'CPU_registered_after':True,'accelerator_devices_after':0}


def capture():
    return {'state_parameters':{},'native_backend_policy':registry(),'native_linkage':{},'native_resource_context':{},
        'diagnostic':True,'model_loads':1,'physical_slots':1,'threads':10,'gpu_layers':0,
        'mode':'probe128','diagnostic_scope':g.DIAGNOSTIC_SCOPE,'complete':False,'status':'probe_only',
        'diagnostic_window_complete':True,'whole_model_qualified':False,'full_campaign_complete':False,
        'generation_requested':False,'generation_performed':False,
        'requests':[{'request':0,'agent_id':1,'prompt_tokens':521,'computed_tokens':128,'next_position':128,
            'full_prompt_computed':False,'status':'probe_only','complete':False,
            'logits_requested_at_probe_end':False,'continuation':[],'restored_prefix_tokens':0,
            'prompt_ids_sha256':'a'*64}]}


def document():
    fields=('provider_patch_sha256','kernel_sha256','runner_source_sha256','native_guard_policy_sha256',
        'helper_source_sha256','snapshot_source_sha256','state_capture_source_sha256',
        'contract_source_sha256','profile_source_sha256','provider_build_receipt_sha256',
        'cpu_geometry_source_sha256','cpu_geometry_manifest_sha256')
    return g.profiles.profile({'CPU_METAL_Q8_BRIDGE_COMPILED':True,
        'native_libraries_sha256':{'ggml-cpu':'a'*64,'ggml-base':'b'*64,'llama':'c'*64},
        **{field:'d'*64 for field in fields}})


class ProbeNegatives(unittest.TestCase):
    def validate_capture(self,value):
        metadata={'snapshot_manifest':{'state_parameters':{}},'request0_full_native_ids_sha256':'a'*64}
        # These tests isolate the NEW scope boundary. Existing linkage/profile/
        # baseline checks are independently exercised by the source package tests.
        with patch.object(g,'validate_state'),patch.object(g,'validate_loaded_images'),patch.object(g,'validate_resource_context'):
            return g.validate_capture_report(value,metadata,{},'.',0,32)

    def test_only_incomplete_full_prompt_can_finish_probe_window(self):
        self.assertTrue(self.validate_capture(capture())['passed'])

    def test_full_completion_claim_rejected(self):
        value=capture();value['complete']=True
        with self.assertRaises(g.GpuContractError):self.validate_capture(value)

    def test_truncated_prompt_identity_rejected(self):
        value=capture();value['requests'][0]['prompt_tokens']=128
        with self.assertRaises(g.GpuContractError):self.validate_capture(value)

    def test_changed_full_ids_rejected(self):
        value=capture();value['requests'][0]['prompt_ids_sha256']='b'*64
        with self.assertRaises(g.GpuContractError):self.validate_capture(value)

    def test_artificial_boundary_logits_rejected(self):
        value=capture();value['requests'][0]['logits_requested_at_probe_end']=True
        with self.assertRaises(g.GpuContractError):self.validate_capture(value)

    def test_generation_rejected(self):
        value=capture();value['generation_performed']=True
        with self.assertRaises(g.GpuContractError):self.validate_capture(value)

    def test_blas_or_other_accel_rejected(self):
        for field,val in [('registry_after',['MTL','CPU','BLAS']),('accelerator_devices_after',1),('CPU_registered_after',False)]:
            value=registry();value[field]=val
            with self.subTest(field=field),self.assertRaises(g.GpuContractError):g.validate_backend_policy(value)

    def test_no_relabeling_old_tail_profile(self):
        value=registry();value['policy']='legacy'
        with self.assertRaises(g.GpuContractError):g.validate_backend_policy(value)

    def test_warning_requires_original_headroom_plus_gpu_reservation(self):
        sample={'pressure_level':2,'available_bytes':(768+128+512)*(1<<20),'rss_bytes':1,'swap_growth_bytes':0}
        self.assertTrue(g.sufficient_memory_sample(sample,0))
        sample['available_bytes']-=1
        self.assertFalse(g.sufficient_memory_sample(sample,0))

    def test_original_hardcaps_remain(self):
        sample={'pressure_level':1,'available_bytes':2<<30,'rss_bytes':int(8.5*(1<<30)),'swap_growth_bytes':32<<20}
        self.assertTrue(g.sufficient_memory_sample(sample,0))
        for field in ('rss_bytes','swap_growth_bytes'):
            value=copy.deepcopy(sample);value[field]+=1
            with self.subTest(field=field):self.assertFalse(g.sufficient_memory_sample(value,0))

    def test_missing_or_nonfinite_sample_fails_closed(self):
        for sample in ({},{'pressure_level':2,'available_bytes':float('nan'),'rss_bytes':1,'swap_growth_bytes':0}):
            self.assertFalse(g.sufficient_memory_sample(sample,0))

    def test_profile_without_commands_is_not_execution(self):
        doc=document()
        text=('AIRLLM_CPU_METAL_Q8 enabled=1 backend='+g.BACKEND+
            ' canonical_sha256='+doc['canonical_sha256']+' kernel_sha256='+'d'*64+'\n'+
            'AIRLLM_CPU_METAL_Q8 allocation_count=3 gpu_payload_current=4751360 gpu_payload_peak=4751360 '
            'weight_tile_bytes=35651584 native_swap_baseline_bytes=0\n')
        with self.assertRaises(g.GpuContractError) as caught:g.parse_trace(text,doc,baseline_swap=0)
        self.assertEqual(caught.exception.code,'GPU_ZERO_COMMANDS')

    def test_profile_requires_new_helper_fingerprints(self):
        fingerprints=copy.deepcopy(document()['canonical_identity']['fingerprints'])
        del fingerprints['helper_source_sha256']
        with self.assertRaises(ValueError):g.profiles.profile(fingerprints)


if __name__=='__main__':unittest.main()
