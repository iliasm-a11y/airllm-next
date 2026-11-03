"""Metadata unit tests and bounded source checks; no native/GPU/build execution."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bridge_profile', ROOT/'profile.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def identities():
    return {'CPU_METAL_Q8_BRIDGE_COMPILED': True,
            'native_libraries_sha256': {'ggml-cpu':'1'*64,'ggml-base':'2'*64,'llama':'3'*64},
            'provider_patch_sha256':'4'*64,'kernel_sha256':'5'*64,
            'runner_source_sha256':'6'*64,'native_guard_policy_sha256':'7'*64}


class ProfileTests(unittest.TestCase):
    def test_gpu_identity_even_with_zero_gpu_layers(self):
        p = module.profile(identities())
        self.assertTrue(p['canonical_identity']['gpu_execution_possible'])
        self.assertEqual(p['canonical_identity']['n_gpu_layers'], 0)
        self.assertEqual(p['canonical_identity']['backend'], 'native_cpu_with_metal_q8_bridge')
        self.assertEqual(p['environment']['AIRLLM_CPU_METAL_Q8_CANONICAL_SHA256'], p['canonical_sha256'])
        self.assertEqual(p['render_options_extension']['gpu_execution_profile']['gpu_policy_sha256'], p['canonical_sha256'])

    def test_library_identity_changes_key(self):
        a, b = identities(), identities()
        b['native_libraries_sha256']['ggml-cpu'] = '8'*64
        self.assertNotEqual(module.profile(a)['canonical_sha256'], module.profile(b)['canonical_sha256'])

    def test_extension_is_accepted_inside_existing_snapshot_schema(self):
        key_file = ROOT.parent/'airllm-next/research/glm53_batched_prefill/snapshot_key.py'
        key_spec = importlib.util.spec_from_file_location('bridge_snapshot_key',key_file)
        keys = importlib.util.module_from_spec(key_spec); key_spec.loader.exec_module(keys)
        params = dict.fromkeys(keys.PARAMETERS,0)
        params.update(n_ctx=2048,n_batch=16,n_ubatch=16,n_seq_max=1,n_threads=10,n_threads_batch=10)
        manifest = {'schema':1,'model_shards':[{'sha256':'1'*64,'size_bytes':1}],
            'tokenizer_sha256':'2'*64,'template_sha256':'3'*64,'runtime_source_sha256':'4'*64,
            'runtime_binaries':{'helper':'5'*64},'prefix_tokens':[1,2], 'prefix_positions':[0,1],
            'state_parameters':params,'render_options':{'native_environment':{}}}
        before = keys.canonical_key(manifest)[0]
        p = module.profile(identities())
        manifest['render_options'].update(p['render_options_extension'])
        manifest['render_options']['native_environment'].update(p['environment'])
        self.assertNotEqual(before,keys.canonical_key(manifest)[0])

    def test_refuses_original_uncompiled_runtime(self):
        for flag in (False, 1, None):
            data = identities(); data['CPU_METAL_Q8_BRIDGE_COMPILED'] = flag
            with self.assertRaises(ValueError): module.profile(data)

    def test_refuses_missing_provenance(self):
        for key in ('runner_source_sha256','native_guard_policy_sha256','provider_patch_sha256'):
            data = identities(); del data[key]
            with self.assertRaises(ValueError): module.profile(data)

    def test_refuses_malformed_or_extra_metadata(self):
        for key, value in (('kernel_sha256','A'*64), ('native_libraries_sha256',{}), ('unexpected','private/path')):
            data = identities(); data[key] = value
            with self.assertRaises(ValueError): module.profile(data)


def source_checks():
    native = ROOT/'source/ggml/src/ggml-cpu'
    source = (native/'metal-q8-bridge.mm').read_text()
    cmake = (native/'CMakeLists.txt').read_text()
    cpp = (native/'ggml-cpu.cpp').read_text()
    kernel = (native/'metal-q8-bridge-kernel.h').read_text()
    embedded = kernel.split('R"Q8BRIDGE(',1)[1].rsplit(')Q8BRIDGE";',1)[0]
    declared = re.search(r'KERNEL_SHA256 "([0-9a-f]{64})"',kernel).group(1)
    checks = {
        'build_default_off': 'option(GGML_CPU_METAL_Q8_BRIDGE "Build experimental bounded CPU-to-Metal Q8 provider" OFF)' in cmake,
        'registration_compiled_guard': '#ifdef GGML_CPU_METAL_Q8_BRIDGE\n        if (auto * hook =' in cpp,
        'runtime_default_off': 'if (!flag || std::strcmp(flag,"1") != 0) return false;' in source,
        'canonical_gpu_identity_required': 'AIRLLM_CPU_METAL_Q8_CANONICAL_SHA256' in source,
        'embedded_kernel_exact_sha': hashlib.sha256(embedded.encode()).hexdigest() == declared,
        'native_cpu_quantizer': 'ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float' in source,
        'cpu_mmap_pointer_identity_helper': 'ggml_backend_cpu_buffer_has_plain_layout(buffer)' in source,
        'fixed_allocations_only': source.count('newBufferWithLength:') == 4 and 'newBufferWithLength:' not in source.split('bool run(',1)[1],
        'no_mapped_weight_placement': 'newBufferWithBytesNoCopy' not in source and 'tensor_buft_overrides' not in source,
        'all_workers_snapshot_admission': bool(re.search(r'const bool admitted = shared->decision == ADMITTED;[\s\S]*?ggml_barrier\(params->threadpool\);\s*if \(!admitted\) return false;',source)),
        'all_workers_snapshot_completion': bool(re.search(r'const bool handled = shared->decision == COMPLETED;\s*ggml_barrier\(params->threadpool\);[^\n]*\n\s*return handled;',source)),
        'single_final_dst_publication': source.count('std::memcpy(op->data, staged, ggml_nbytes(op));') == 1 and 'if (workspace->run(op, quant, staged)) {' in source,
        'fatal_pending_timeout': 'timeout pending_gpu=1 output_unpublished=1; no CPU fallback' in source,
        'fatal_late_scale_refusal': 'if (submitted_this_op) GGML_ABORT' in source,
        'actual_shared_cpu_workspace_counted': 'params->wsize > DATA_LIMIT - GPU_CAP' in source,
        'same_native_swap_baseline': 'baseline_swap = initial_swap_baseline();' in source and 'AIRLLM_CPU_METAL_Q8_SWAP_BASELINE_BYTES' in source,
        'pressure_two_sufficient_subset': '(s.pressure != 1 && s.pressure != 2)' in source and 'RESERVE_BYTES + GPU_RESERVATION_BYTES + (s.pressure == 2 ? WARNING_HEADROOM_BYTES : 0)' in source,
        'separate_device_allocation_measurement': 'device.currentAllocatedSize' in source and 'DEVICE_ALLOCATION_LIMIT = 64u << 20' in source and 'unmeasured_driver_residency=null' in source,
        'tails_6_14_not_whitelisted': 'return n == 8 || n == 16 || n == 32 || n == 64 || n == 128;' in source,
    }
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(),filename=str(path))
    for path in (ROOT/'source').rglob('*'):
        if path.is_file() and any(line.rstrip()!=line for line in path.read_text().splitlines()):
            raise AssertionError('trailing whitespace: '+str(path))
    if not all(checks.values()): raise AssertionError(checks)
    # Source-derived byte arithmetic, not execution of the C++ allocator.
    gpu = 1536*4096//32*34 + 128*4096 + 128*4096//32*2 + 128*1536*4
    quant = 128*4096//32*34
    staged = (64+quant+63)&~63
    host = staged+8192*128*4
    assert gpu == 8028160 and host == 4751424 and gpu+host < 64<<20
    for blocks in (64,128):
        for threads in (1,8,10,64,128):
            intervals=[(i*blocks//threads,(i+1)*blocks//threads) for i in range(threads)]
            assert intervals[0][0] == 0 and intervals[-1][1] == blocks
            assert all(intervals[i][1] == intervals[i+1][0] for i in range(threads-1))
    return {'checks':checks,'maximum_provider_cpu_bytes':host,'fixed_gpu_bytes':gpu,
            'maximum_provider_logical_total_bytes':host+gpu,
            'limitations':['no C++/Objective-C++ compilation','no native threading, allocator or GPU execution','not a semantic or fidelity proof']}


if __name__ == '__main__':
    tests = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProfileTests))
    result = source_checks()
    result['metadata_unit_tests']={'run':tests.testsRun,'passed':tests.wasSuccessful()}
    (ROOT/'static-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'metadata_unit_tests':tests.testsRun,'source_checks':len(result['checks']),
                      'passed':tests.wasSuccessful(),'gpu_calls':0,'build_calls':0}))
    raise SystemExit(0 if tests.wasSuccessful() else 1)
