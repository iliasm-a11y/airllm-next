"""Distinct noBLAS/fenced128 GPU diagnostic identity; never launches anything."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

NATIVE_MEMORY_POLICY = {
    'reserve_bytes':768<<20, 'warning_headroom_bytes':512<<20,
    'warning_swap_growth_bytes':32<<20, 'max_swap_growth_bytes':32<<20,
    'max_rss_bytes':int(8.5*(1<<30)), 'headroom_grace_seconds':3,
    'warning_grace_seconds':20,
}

def profile(fingerprints):
    if fingerprints.get('CPU_METAL_Q8_BRIDGE_COMPILED') is not True:
        raise ValueError('new backend compiled-feature receipt is required')
    required = ('native_libraries_sha256', 'provider_patch_sha256', 'kernel_sha256',
                'runner_source_sha256', 'native_guard_policy_sha256',
                'helper_source_sha256','snapshot_source_sha256','state_capture_source_sha256',
                'contract_source_sha256','profile_source_sha256','provider_build_receipt_sha256',
                'cpu_geometry_source_sha256','cpu_geometry_manifest_sha256')
    if set(fingerprints) - set(required) - {'CPU_METAL_Q8_BRIDGE_COMPILED'}:
        raise ValueError('only explicit identity fields are accepted')
    for key in required:
        if key not in fingerprints:
            raise ValueError('missing identity: ' + key)
    libraries = fingerprints['native_libraries_sha256']
    if type(libraries) is not dict or not all(name in libraries for name in ('ggml-cpu', 'ggml-base', 'llama')):
        raise ValueError('CPU, base and llama library fingerprints are required')
    hashes = list(libraries.values()) + [fingerprints[k] for k in required[1:]]
    if any(
            type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
            for value in hashes):
        raise ValueError('complete lowercase SHA256 fingerprints required')
    identity = {
        'schema_version': 2, 'identity_domain':'glm53-noblas-nocopy-fenced-probe128-v1',
        'prefill_chunk_policy':'cpu8-tail-global-fences208-no-blas-v4-nocopy-probe128-v1',
        'prefill_chunk_policy_version':1,'prefill_fences':[128,208,256,384,512],
        'fallback_column_chunk':16,'fallback_remainder_policy':'floor16-then-cpu8-odd-tail',
        'tail_reference_microbatch':8,'cpu_accelerator_policy':'unregister-BLAS-before-model-v1',
        'diagnostic_scope':'one_prompt_128_native_tokens_no_generation',
        'full_campaign_qualified':False,'model_numeric_qualified':False,
        'complete_frozen_prompts_sha256':'f2b40d0c9f94a507ccaaff5e131fb57e698585ba640c0614fe2e69c12bd11fef',
        'required_original_agent_ids':{'first':1,'last':600,'unique_count':600},
        'backend': 'native_cpu_with_metal_q8_bridge',
        'gpu_execution_possible': True, 'n_gpu_layers': 0,
        'weights': 'original CPU mmap; one bounded original read-only NoCopy view; same pointer; no permanent Metal weight placement',
        'activation_quantizer': 'native CPU Q8_0 from_float', 'kernel': 'experimental Metal4 MPP int8xint8 int32 dots32 ordered FP32 FMA',
        'admitted_dimensions': {'K': [2048, 4096], 'M_multiple': 8, 'M_max': 8192, 'N': [8,16,32,64,128]},
        'unsupported_shapes': 'CPU; includes N6/14 and single-token decode',
        'native_compute_reference_mode': 'CPU; bridge disabled when use_ref is true',
        'allocation': {'total_limit_bytes': 64 << 20, 'weight_tile_rows': 8192,
                       'gpu_fixed_buffer_bytes': 4751360, 'fixed_buffer_count':3, 'temporary_weight_view_count_max':1, 'borrowed_view_max_bytes':35782656, 'metal_device_current_allocated_limit_bytes':64<<20, 'cpu_staging': 'shared graph workspace; view charged separately to64MiB workspace admission'},
        'resource_policy': {'kind':'conservative_sufficient_native_contextual_subset_v1',
            'native_limits':dict(NATIVE_MEMORY_POLICY),'additional_reservation_bytes':128<<20,
            'admitted_pressure_levels':[1,2],'available_formula':'psutil7.2.2_macos_free_plus_inactive',
            'swap_baseline':'same initial global sample as authoritative native watchdog'},
        'fatal_after_any_gpu_submit': True, 'publish_only_after_all_tiles_terminal_success': True,
        'requires_existing_native_watchdog': True, 'whole_model_qualified': False,
        'requires_runtime_trace_nonzero_gpu_commands': True,
        'fingerprints': fingerprints,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    return {'canonical_identity': identity, 'canonical_sha256': digest,
            'render_options_extension': {'gpu_execution_profile': {
                'schema': 1, 'actual_backend': identity['backend'],
                'gpu_policy_sha256': digest, 'identity': identity}},
            'environment': {'AIRLLM_CPU_METAL_Q8_BRIDGE': '1',
                            'AIRLLM_CPU_METAL_Q8_CANONICAL_SHA256': digest,
                            'AIRLLM_GLM53_CPU_NO_BLAS':'1'},
            'status': 'identity only; no inherited model or performance qualification'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fingerprints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(profile(json.loads(args.fingerprints.read_text())), indent=2) + '\n')
