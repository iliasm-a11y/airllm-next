"""Fresh synthetic metadata rejection tests; never alter measured receipts.

No full comparison is mocked to pass. A real candidate build is intentionally
absent. Dependency-only fixtures test each refusal independently of model data.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fenced208_variant_compare',ROOT/'variant_compare.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def fixture():
    """Entirely synthetic dependency metadata, not an inference/build receipt."""
    binding={'reference':{'binary_sha256':v.REFERENCE_BINARY,'helper_source_sha256':v.REFERENCE_SOURCE},
        'candidate':{'binary_sha256':'1'*64,'helper_source_sha256':v.CANDIDATE_SOURCE},
        'shared_source_sha256':{'snapshot_source':'2'*64,'state_capture_source':'3'*64,'native_manifest':'4'*64,'wrapper_source':'5'*64},
        'shared_native_libraries_sha256':{'libfixture.dylib':'6'*64}}
    params={'n_ctx':2048,'n_batch':16,'n_ubatch':16,'n_seq_max':1,'type_k':1,'type_v':1,'flash_attn':-1,
        'offload_kqv':False,'n_gpu_layers':0,'rope_scaling_type':-1,'rope_freq_base':0,'rope_freq_scale':0,'n_threads':10,'n_threads_batch':10,**v.POLICY_PARAMETERS}
    sources={**binding['shared_source_sha256'],'helper_source':v.CANDIDATE_SOURCE}
    binaries={'helper':'1'*64,**binding['shared_native_libraries_sha256']}
    manifest={'schema':1,'model_shards':[{'sha256':'7'*64,'size_bytes':10}],
        'tokenizer_sha256':'8'*64,'template_sha256':'9'*64,
        'runtime_source_sha256':hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'runtime_binaries':binaries,'prefix_tokens':list(range(208)),'prefix_positions':list(range(208)),
        'state_parameters':params,'render_options':{'fixture_only':True}}
    report={'microbatch':16,'logical_batch':16,'state_parameters':params,'prefill_chunk_policy':v.POLICY,
        'prefill_chunk_policy_qualified':False,'tail_reference_microbatch':8}
    receipt={'scope':'synthetic dependency unit fixture only','preflight':{'snapshot_manifest':manifest,'source_sha256':sources,'runtime_binaries':binaries}}
    build={'parameter_smoke':{'16':{'returncode':0,'stdout':json.dumps(params),'state_parameters':params}}}
    reset_key(report,receipt)
    return report,receipt,build,{'binding':binding}

def reset_key(report,receipt):
    key,_=v.canonical_key(receipt['preflight']['snapshot_manifest'])
    report['snapshot_key_sha256']=key;receipt['preflight']['snapshot_key_sha256']=key

class RejectionTests(unittest.TestCase):
    def assert_bad(self,data):
        report,receipt,build,pair=data
        with self.assertRaises(ValueError):v.arm_dependencies(report,receipt,build,pair,'candidate')
    def test_unbuilt_binding_never_reaches_a_full_capture_comparison(self):
        with self.assertRaises(ValueError):
            v.load_build_pair(ROOT/'does-not-exist-reference',ROOT/'does-not-exist-candidate',ROOT/'binding-template.json')
    def test_fake_build_flag_rejected_even_when_its_file_hash_is_self_consistent(self):
        prepared=v.read(ROOT/'prepare-manifest.json');binding=v.read(ROOT/'binding-template.json')
        binding['status']='built_bound_not_numerically_qualified';binding['candidate']['binary_sha256']='1'*64
        fake={'returncode':0,'status':'built_metadata_only_smoked','build_performed':False,'binary_rebuilt':True,
            'runtime_hashes_verified_for_this_build':True,'model_loaded':False,'gpu_commands':0,'policy':v.POLICY,
            'original_build_receipt_sha256':binding['reference']['build_receipt_sha256']}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'fake-build.json';p.write_text(json.dumps(fake))
            binding['candidate']['build_receipt_sha256']=v.sha(p)
            bp=root/'fake-binding.json';bp.write_text(json.dumps(binding))
            with self.assertRaises(ValueError) as raised:v.load_build_pair(prepared['original_build_receipt'],p,bp)
            self.assertIn('candidate build/metadata provenance',str(raised.exception))
    def test_wrong_policy_is_rejected_with_recomputed_own_key(self):
        data=fixture();report,receipt,build,pair=data
        report['prefill_chunk_policy']='wrong';report['state_parameters']['prefill_chunk_policy']='wrong'
        build['parameter_smoke']['16']['stdout']=json.dumps(report['state_parameters']);reset_key(report,receipt)
        self.assert_bad(data)
    def test_wrong_helper_binary_and_native_library_are_rejected(self):
        for name in ('helper','libfixture.dylib'):
            data=fixture();report,receipt,_,_=data
            receipt['preflight']['snapshot_manifest']['runtime_binaries'][name]='0'*64
            reset_key(report,receipt);self.assert_bad(data)
    def test_wrong_helper_or_wrapper_source_is_rejected(self):
        for name in ('helper_source','wrapper_source'):
            data=fixture();report,receipt,_,_=data
            sources=receipt['preflight']['source_sha256'];sources[name]='0'*64
            receipt['preflight']['snapshot_manifest']['runtime_source_sha256']=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            reset_key(report,receipt);self.assert_bad(data)
    def test_stale_key_and_smoke_mismatch_are_rejected(self):
        data=fixture();data[0]['snapshot_key_sha256']='0'*64;self.assert_bad(data)
        data=fixture();data[2]['parameter_smoke']['16']['stdout']='{}';self.assert_bad(data)

if __name__=='__main__':unittest.main()
