"""Pure validation for the isolated GPU runner; no native library is loaded."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BACKEND = 'native_cpu_with_metal_q8_bridge'
GPU_BYTES = 4751360
WEIGHT_TILE_BYTES = 35651584
DATA_LIMIT = 64 << 20
ADDITIONAL_RESERVATION = 128 << 20  # data64 + framework/driver policy64, not measured RSS
TAIL_POLICY = 'cpu8-tail-global-fences208-no-blas-v4-nocopy-probe128-v1'
CPU_ACCELERATOR_POLICY='unregister-BLAS-before-model-v1'
FENCES=[128,208,256,384,512]
DIAGNOSTIC_SCOPE='one_prompt_128_native_tokens_no_generation'
FROZEN_PROMPTS_SHA256='f2b40d0c9f94a507ccaaff5e131fb57e698585ba640c0614fe2e69c12bd11fef'
spec = importlib.util.spec_from_file_location('cpu_metal_q8_profile', HERE/'profile.py')
profiles = importlib.util.module_from_spec(spec); spec.loader.exec_module(profiles)


class GpuContractError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def require(condition, code, message):
    if not condition: raise GpuContractError(code, message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): digest.update(block)
    return digest.hexdigest()


def read_json(path, limit=4<<20):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= limit, 'GPU_METADATA_SIZE', 'Bounded existing JSON evidence required')
    try:return json.loads(path.read_text())
    except (ValueError,UnicodeError) as error:raise GpuContractError('GPU_METADATA_PARSE',str(error)) from error


def validate_full_prompts(document, actual_json_sha256):
    """Require the reviewed byte-identical complete input, without invented fields."""
    require(actual_json_sha256==FROZEN_PROMPTS_SHA256,'GPU_PROMPTS_IDENTITY','The complete frozen native prompt JSON differs')
    require(type(document) is dict and type(document.get('agents')) is list and len(document['agents'])==600
            and type(document.get('template')) is str and bool(document['template']),
            'GPU_PROMPTS_COMPLETE','All600 original agents and the native template are required')
    agents={}
    for row in document['agents']:
        require(type(row) is dict and type(row.get('agent_id')) is int and 1<=row['agent_id']<=600
                and row['agent_id'] not in agents,'GPU_PROMPTS_COMPLETE','Exactly unique integer agent IDs1..600 required')
        tokens=row.get('token_ids')
        require(type(tokens) is list and 208<len(tokens)<=1800
                and all(type(token) is int and 0<=token<2**31 for token in tokens),
                'GPU_PROMPTS_COMPLETE','Complete bounded native token ID lists required for all600 agents')
        agents[row['agent_id']]=row
    require(set(agents)==set(range(1,601)),'GPU_PROMPTS_COMPLETE','Original IDs1..600 are required')
    prefix=agents[1]['token_ids'][:208]
    require(all(row['token_ids'][:208]==prefix for row in agents.values()),
            'GPU_PROMPTS_COMPLETE','All600 common native prefixes must remain identical')
    return agents


def validate_profile(document):
    try: expected = profiles.profile(document['canonical_identity']['fingerprints'])
    except (KeyError, TypeError, ValueError) as error:
        raise GpuContractError('GPU_PROFILE_INVALID',str(error)) from error
    require(document == expected, 'GPU_PROFILE_IDENTITY', 'Profile body, backend, key or environment differs from its canonical identity')
    return document


def validate_limits(limits, document):
    expected=document['canonical_identity']['resource_policy']['native_limits']
    require(type(limits) is dict and all(type(limits.get(key)) in (int,float)
            and type(limits.get(key)) is not bool and math.isfinite(limits[key]) and limits[key] == value
            for key,value in expected.items()),
            'GPU_MEMORY_POLICY','This pinned provider requires the unchanged approved native memory limits')


def sufficient_memory_sample(sample, baseline_swap):
    """Conservative sufficient domain, never inference about grace timestamps."""
    keys=('pressure_level','available_bytes','rss_bytes','swap_growth_bytes')
    if type(sample) is not dict or any(type(sample.get(key)) is not int or sample[key] < 0 for key in keys):return False
    if type(baseline_swap) is not int or baseline_swap < 0:return False
    policy=profiles.NATIVE_MEMORY_POLICY
    pressure=sample['pressure_level']
    floor=policy['reserve_bytes']+ADDITIONAL_RESERVATION+(policy['warning_headroom_bytes'] if pressure==2 else 0)
    return pressure in (1,2) and sample['available_bytes'] >= floor and sample['rss_bytes'] <= policy['max_rss_bytes'] \
        and sample['swap_growth_bytes'] <= min(policy['max_swap_growth_bytes'],policy['warning_swap_growth_bytes'] if pressure==2 else policy['max_swap_growth_bytes'])


def validate_resource_context(context, baseline_swap):
    require(type(context) is dict and context.get('schema') == 1 and context.get('gpu_bridge_requested') is True
            and type(context.get('native_guard_swap_baseline_bytes')) is int
            and context['native_guard_swap_baseline_bytes'] == baseline_swap,
            'GPU_RESOURCE_BASELINE','Native bridge and external watchdog swap baseline differ')


def validate_state(state, document, microbatch):
    validate_profile(document)
    require(state.get('cpu_metal_q8_bridge_compiled') is True, 'GPU_RUNTIME_NOT_COMPILED', 'Runtime feature does not contain the provider')
    require(state.get('execution_backend') == BACKEND and state.get('gpu_execution_possible') is True
            and state.get('gpu_bridge_requested') is True, 'GPU_BACKEND_MISMATCH', 'Native helper must identify actual GPU-capable execution explicitly')
    require(state.get('gpu_policy_sha256') == document['canonical_sha256'], 'GPU_PROFILE_IDENTITY', 'Native helper profile key differs')
    require(type(state.get('n_gpu_layers')) is int and state['n_gpu_layers'] == 0 and state.get('op_offload') is False and state.get('offload_kqv') is False
            and type(state.get('moe_cache_size')) is int and state['moe_cache_size'] == 0
            and type(state.get('permanent_gpu_weight_bytes')) is int and state['permanent_gpu_weight_bytes'] == 0
            and state.get('weight_storage_policy') == 'cpu_originals_nocopy_view',
            'GPU_WEIGHT_PLACEMENT', 'Permanent or opaque GPU weight placement is forbidden for this bridge')
    require(all(type(state.get(key)) is int for key in ('n_ctx','n_seq_max','n_batch','n_ubatch'))
            and state.get('n_ctx') == 2048 and state.get('n_seq_max') == 1 and state.get('n_batch') == microbatch
            and state.get('n_ubatch') == microbatch, 'GPU_STATE_PARAMETERS', 'Native state descriptor differs')
    require(state.get('prefill_chunk_policy') == TAIL_POLICY and state.get('prefill_chunk_policy_version') == 1
            and state.get('tail_reference_microbatch') == 8
            and state.get('prefill_fences')==FENCES
            and state.get('fallback_column_chunk')==16 and state.get('fallback_remainder_policy')=='floor16-then-cpu8-odd-tail'
            and state.get('cpu_accelerator_policy')==CPU_ACCELERATOR_POLICY
            and state.get('diagnostic_scope')==DIAGNOSTIC_SCOPE, 'GPU_TAIL_POLICY', 'The isolated helper must preserve the CPU8 odd-tail policy')


def validate_backend_policy(observed):
    require(type(observed) is dict and observed.get('policy')==CPU_ACCELERATOR_POLICY,
            'GPU_BACKEND_REGISTRY','Explicit noBLAS observation required')
    before,after=observed.get('registry_before'),observed.get('registry_after')
    require(type(before) is list and type(after) is list and len(before)<=16 and len(after)<=16
            and all(type(name) is str for name in before+after)
            and len(set(before))==len(before) and len(set(after))==len(after)
            and 'BLAS' in before and 'CPU' in before and 'BLAS' not in after and 'CPU' in after
            and set(after)==set(before)-{'BLAS'}
            and observed.get('BLAS_registered_after') is False and observed.get('CPU_registered_after') is True
            and type(observed.get('accelerator_devices_after')) is int and observed['accelerator_devices_after']==0,
            'GPU_BACKEND_REGISTRY','CPU required, BLAS removed, and zero ACCEL required')
    return {'passed':True,'CPU_present':True,'BLAS_removed':True,'ACCEL_devices':0}


OVERLAY_FILES=('ggml/include/ggml-backend.h','ggml/src/ggml-backend.cpp','ggml/src/ggml-cpu/CMakeLists.txt',
    'ggml/src/ggml-cpu/ggml-cpu.cpp','ggml/src/ggml-cpu/metal-q8-bridge.h',
    'ggml/src/ggml-cpu/metal-q8-bridge.mm','ggml/src/ggml-cpu/metal-q8-bridge-kernel.h')
HEADERS=('include/llama.h','ggml/include/ggml.h','ggml/include/ggml-backend.h','vendor/nlohmann/json.hpp')
V4_BUILD_SHA='7b0ac6dddabf542aae42d7fb8bb8762d6b98e1705dc423b09ad071112d0c8750'

def validate_input_provenance(receipt, required_roles):
    roles=receipt.get('input_roles');tracked=receipt.get('inputs_sha256')
    require(receipt.get('schema_version') == 2 and type(roles) is dict and set(required_roles) <= set(roles)
            and type(tracked) is dict and receipt.get('inputs_before') == tracked == receipt.get('inputs_after'),
            'GPU_BUILD_PROVENANCE','Complete source/include roles and identical before/after build hashes required')
    require(all(type(value) is str and Path(value).is_absolute() and value in tracked for value in roles.values()),
            'GPU_BUILD_PROVENANCE','Each declared role must identify an explicitly tracked input')
    require(all(type(path) is str and Path(path).is_absolute() and type(value) is str
            and re.fullmatch('[0-9a-f]{64}',value) and Path(path).is_file() and sha(path)==value
            for path,value in tracked.items()),'GPU_BUILD_PROVENANCE','Tracked build input differs on disk')
    return roles


def validate_artifacts(document, build_receipt, driver_receipt, *, libraries, binary, helper_source, runner_source, guard_source, provider_receipt_sha256):
    validate_profile(document)
    fingerprints = document['canonical_identity']['fingerprints']
    require(build_receipt.get('status') == 'built' and build_receipt.get('CPU_METAL_Q8_BRIDGE_COMPILED') is True,
            'GPU_BUILD_RECEIPT', 'A completed provider build receipt is required')
    provider_roles=validate_input_provenance(build_receipt,
        ('source_manifest','bridge_patch','reconstruction_receipt','v4_build_manifest','build_script','kernel_standalone',
         *('overlay:'+name for name in OVERLAY_FILES),*('native_overlay:'+name for name in OVERLAY_FILES),
         *('header:'+name for name in HEADERS),'cmake:CMakeLists.txt','cmake:ggml/CMakeLists.txt'))
    require(sha(provider_roles['v4_build_manifest'])==V4_BUILD_SHA,'GPU_BUILD_PROVENANCE','Actual pinned V4 build manifest required')
    v4=read_json(provider_roles['v4_build_manifest'])
    reconstruction=read_json(provider_roles['reconstruction_receipt'])
    require(reconstruction.get('status')=='source_exact' and len(reconstruction.get('source_checks',{}))==13
            and reconstruction.get('build_manifest_sha256')==V4_BUILD_SHA,
            'GPU_BUILD_PROVENANCE','Exact13-source V4 reconstruction receipt required')
    required=tuple('pin:'+row['name'] for row in v4['patches'])+tuple('native:'+name for name in reconstruction['source_checks'])
    validate_input_provenance(build_receipt,required)
    overlay=read_json(provider_roles['source_manifest']).get('files',{})
    require(set(overlay)==set(OVERLAY_FILES),'GPU_BUILD_PROVENANCE','Exactly seven declared overlay files required')
    for name,row in overlay.items():
        expected=row.get('candidate_sha256')
        require(sha(provider_roles['overlay:'+name])==expected==sha(provider_roles['native_overlay:'+name]),
                'GPU_BUILD_PROVENANCE','Staged and original overlay inputs differ')
    for name,row in reconstruction['source_checks'].items():
        expected=overlay.get(name,{}).get('candidate_sha256',row.get('expected_sha256'))
        require(row.get('matches') is True and sha(provider_roles['native:'+name])==expected,
                'GPU_BUILD_PROVENANCE','Staged native V4 source differs')
    for row in v4['patches']:
        require(sha(provider_roles['pin:'+row['name']])==row['sha256'],'GPU_BUILD_PROVENANCE','V4 patch pin differs')
    require(sha(provider_roles['bridge_patch'])==fingerprints['provider_patch_sha256']
            and sha(provider_roles['kernel_standalone'])==fingerprints['kernel_sha256']
            and build_receipt.get('compiled_feature_query_verified') is True,
            'GPU_BUILD_IDENTITY','Patch, kernel and actual feature-query evidence required')
    for field in ('provider_patch_sha256','kernel_sha256'):
        require(build_receipt.get(field) == fingerprints[field], 'GPU_BUILD_IDENTITY', 'Provider build fingerprint differs: '+field)
    require(build_receipt.get('native_libraries_sha256') == fingerprints['native_libraries_sha256'],
            'GPU_BUILD_IDENTITY', 'Provider build library fingerprints differ')
    actual = {}
    for name, expected in fingerprints['native_libraries_sha256'].items():
        require(type(name) is str and re.fullmatch(r'[A-Za-z0-9_.-]+',name) is not None,
                'GPU_LIBRARY_IDENTITY','Logical library names only')
        path = Path(libraries)/('lib'+name+'.dylib')
        require(path.is_file() and sha(path) == expected, 'GPU_LIBRARY_IDENTITY', 'Loaded package library differs: '+name)
        actual[name] = expected
    require(sha(runner_source) == fingerprints['runner_source_sha256'], 'GPU_RUNNER_IDENTITY', 'Runner source differs from profile')
    require(sha(guard_source) == fingerprints['native_guard_policy_sha256'], 'GPU_GUARD_IDENTITY', 'Existing guard source differs from profile')
    require(type(driver_receipt.get('returncode')) is int and driver_receipt['returncode'] == 0 and driver_receipt.get('binary_sha256') == sha(binary),
            'GPU_DRIVER_BUILD', 'The isolated GPU helper requires its exact successful build receipt')
    driver_roles=validate_input_provenance(driver_receipt,
        ('helper_source','snapshot_source','state_capture_source','build_script','profile_source','contract_source','runner_source','provider_receipt','cpu_v4_source','cpu_v4_prepare_manifest',*('header:'+name for name in HEADERS)))
    require(driver_receipt.get('status')=='built_metadata_only_smoked'
            and Path(driver_roles['helper_source']).resolve()==Path(helper_source).resolve()
            and driver_receipt.get('native_libraries_sha256')==fingerprints['native_libraries_sha256'],
            'GPU_DRIVER_BUILD','Driver must identify the exact same provider library package and source')
    for ub in (8,16,32,64,128):
        smoke=driver_receipt.get('parameter_smoke',{}).get(str(ub),{})
        require(smoke.get('returncode')==0 and type(smoke.get('stdout')) is str,
                'GPU_DRIVER_SMOKE','Actual native stdout required for every admitted microbatch')
        try:descriptor=json.loads(smoke['stdout'])
        except (ValueError,TypeError) as error:raise GpuContractError('GPU_DRIVER_SMOKE','Malformed actual parameter smoke') from error
        require(type(descriptor) is dict,'GPU_DRIVER_SMOKE','Actual parameter descriptor must be an object')
        linkage=descriptor.pop('native_linkage',None);context=descriptor.pop('native_resource_context',None)
        registry=descriptor.pop('native_backend_policy',None);validate_backend_policy(registry)
        require(descriptor==smoke.get('state_parameters') and linkage==smoke.get('native_linkage')
                and context==smoke.get('native_resource_context') and registry==smoke.get('native_backend_policy'),'GPU_DRIVER_SMOKE','Reserved envelopes or smoke parameters differ')
        validate_state(descriptor,document,ub);validate_loaded_images(linkage,document,libraries);validate_resource_context(context,0)
    require(driver_receipt.get('provider_build_receipt_sha256')==provider_receipt_sha256,
            'GPU_BUILD_PROVENANCE','Driver receipt must bind the exact provider receipt')
    for role,field in (('helper_source','helper_source_sha256'),('snapshot_source','snapshot_source_sha256'),
            ('state_capture_source','state_capture_source_sha256'),('runner_source','runner_source_sha256'),
            ('profile_source','profile_source_sha256'),('contract_source','contract_source_sha256'),
            ('provider_receipt','provider_build_receipt_sha256'),
            ('cpu_v4_source','cpu_geometry_source_sha256'),('cpu_v4_prepare_manifest','cpu_geometry_manifest_sha256')):
        require(sha(driver_roles[role])==fingerprints[field], 'GPU_DRIVER_SOURCE','Profile source/receipt fingerprint differs: '+role)
    require(driver_receipt.get('cpu_accelerator_policy')==CPU_ACCELERATOR_POLICY
            and driver_receipt.get('prefill_fences')==FENCES and driver_receipt.get('diagnostic_scope')==DIAGNOSTIC_SCOPE,
            'GPU_DRIVER_SOURCE','Driver scheduling/noBLAS/probe domain differs')
    recorded = driver_receipt.get('inputs_sha256',{})
    require(type(recorded) is dict and recorded.get(str(Path(helper_source).resolve())) == sha(helper_source),
            'GPU_DRIVER_SOURCE','Helper source does not match its binary build receipt')
    require(driver_receipt.get('prefill_chunk_policy') == TAIL_POLICY,
            'GPU_TAIL_POLICY','GPU driver build must identify the CPU8-compatible odd-tail variant')
    return {'passed':True,'libraries':actual,'provider_build_receipt_verified':True,'driver_build_receipt_verified':True}


def validate_loaded_images(state, document, libraries):
    """Check actual dyld paths, not just files that happen to exist in a package."""
    require(type(state) is dict, 'GPU_LOADED_IMAGES','Explicit native linkage envelope required')
    images = state.get('loaded_native_images')
    require(type(images) is list and 3 <= len(images) <= 64 and all(type(value) is str for value in images) and len(set(images)) == len(images),
            'GPU_LOADED_IMAGES','Unique actual loaded native image paths required')
    expected = document['canonical_identity']['fingerprints']['native_libraries_sha256']
    seen = set()
    for value in images:
        require(type(value) is str and Path(value).is_absolute(), 'GPU_LOADED_IMAGES','Actual absolute dyld image path required')
        path = Path(value).resolve()
        name = path.name
        require(name.startswith(('libggml','libllama')) and name.endswith('.dylib'), 'GPU_LOADED_IMAGES','Unexpected native image name')
        logical = name[3:].split('.',1)[0]
        require(logical in expected and path == (Path(libraries)/('lib'+logical+'.dylib')).resolve()
                and path.is_file() and sha(path) == expected[logical],
                'GPU_LOADED_IMAGES','Loaded native image differs from the fingerprinted package')
        seen.add(logical)
    require({'ggml-cpu','ggml-base','llama'} <= seen, 'GPU_LOADED_IMAGES','Core loaded native images missing')
    return {'passed':True,'actual_loaded_logical_libraries':sorted(seen)}


def validate_capture_report(captured, metadata, document, libraries, baseline_swap, microbatch):
    require(type(captured) is dict, 'GPU_CAPTURE_REPORT', 'Native capture report required')
    expected=metadata['snapshot_manifest']['state_parameters']
    require(type(captured.get('state_parameters')) is dict and captured['state_parameters'] == expected,
            'GPU_STATE_PARAMETERS','Native report, metadata descriptor and key state_parameters must match exactly')
    validate_state(captured['state_parameters'],document,microbatch)
    validate_backend_policy(captured.get('native_backend_policy'))
    linkage=validate_loaded_images(captured.get('native_linkage'),document,libraries)
    validate_resource_context(captured.get('native_resource_context'),baseline_swap)
    requests=captured.get('requests')
    require(captured.get('mode')=='probe128' and captured.get('diagnostic_scope')==DIAGNOSTIC_SCOPE
            and captured.get('diagnostic') is True
            and type(captured.get('model_loads')) is int and captured['model_loads']==1
            and type(captured.get('physical_slots')) is int and captured['physical_slots']==1
            and type(captured.get('threads')) is int and captured['threads']==10
            and type(captured.get('gpu_layers')) is int and captured['gpu_layers']==0
            and captured.get('complete') is False and captured.get('status')=='probe_only' and captured.get('diagnostic_window_complete') is True
            and captured.get('whole_model_qualified') is False and captured.get('full_campaign_complete') is False
            and captured.get('generation_requested') is False and captured.get('generation_performed') is False
            and type(requests) is list and len(requests)==1 and type(requests[0].get('prompt_tokens')) is int
            and 208 < requests[0]['prompt_tokens'] <= 1800 and requests[0].get('full_prompt_computed') is False
            and requests[0].get('status')=='probe_only' and requests[0].get('complete') is False
            and type(requests[0].get('request')) is int and requests[0]['request']==0
            and type(requests[0].get('agent_id')) is int and requests[0]['agent_id']==1
            and requests[0].get('logits_requested_at_probe_end') is False
            and requests[0].get('prompt_ids_sha256')==metadata.get('request0_full_native_ids_sha256')
            and requests[0].get('computed_tokens')==128 and requests[0].get('next_position')==128
            and requests[0].get('continuation')==[] and requests[0].get('restored_prefix_tokens')==0,
            'GPU_DIAGNOSTIC_SCOPE','Only a complete1prompt/128token/no-generation window is admitted')
    return {'passed':True,'state_parameters_exact':True,'native_linkage':linkage,'native_guard_baseline_exact':True}


def parse_trace(text, document, *, baseline_swap):
    """Require real committed/terminal GPU commands and complete final publications."""
    validate_profile(document)
    require(type(baseline_swap) is int and 0 <= baseline_swap <= 2**64-1,'GPU_RESOURCE_BASELINE','Exact initial native swap baseline required')
    require(type(text) is str and len(text.encode()) <= 32<<20, 'GPU_TRACE_SIZE','GPU trace exceeds bounded log size')
    rows = []
    for line in text.splitlines():
        marker = 'AIRLLM_CPU_METAL_Q8 '
        if marker in line:
            body = line.split(marker,1)[1]
            rows.append(dict(re.findall(r'([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)',body)))
    enabled = [row for row in rows if row.get('enabled') == '1']
    require(len(enabled) == 1 and enabled[0].get('backend') == BACKEND
            and enabled[0].get('canonical_sha256') == document['canonical_sha256']
            and enabled[0].get('kernel_sha256') == document['canonical_identity']['fingerprints']['kernel_sha256'],
            'GPU_TRACE_IDENTITY','Missing or inconsistent native GPU identity trace')
    allocations = [row for row in rows if 'gpu_payload_peak' in row]
    require(len(allocations) == 1 and allocations[0].get('allocation_count') == '3'
            and allocations[0].get('gpu_payload_current') == str(GPU_BYTES)
            and allocations[0].get('gpu_payload_peak') == str(GPU_BYTES)
            and allocations[0].get('weight_tile_bytes') == str(WEIGHT_TILE_BYTES)
            and allocations[0].get('native_swap_baseline_bytes') == str(baseline_swap),
            'GPU_ALLOCATION_BOUND','Missing or unbounded fixed allocation evidence')
    submitted = [row for row in rows if row.get('event') == 'submitted']
    completed = [row for row in rows if row.get('event') == 'completed']
    tiles = [row for row in rows if row.get('event') == 'tile_done']
    published = [row for row in rows if row.get('event') == 'published']
    require(bool(submitted) and bool(completed) and bool(published), 'GPU_ZERO_COMMANDS', 'Importing or enabling a profile is not GPU execution evidence')
    def integer(row,key):
        value = row.get(key,'')
        require(re.fullmatch(r'[0-9]+',value) is not None,'GPU_TRACE_COUNTER','Invalid native counter: '+key)
        return int(value)
    def finite_number(row,key):
        try:value=float(row[key])
        except (KeyError,TypeError,ValueError) as error:raise GpuContractError('GPU_TRACE_NUMBER','Missing finite number: '+key) from error
        require(math.isfinite(value) and value >= 0,'GPU_TRACE_NUMBER','Nonfinite/negative native timing')
        return value
    def measured_device_allocation(row):
        current=integer(row,'device_allocated_current')
        peak=integer(row,'device_allocated_observed_peak')
        require(integer(row,'device_allocated_limit') == DATA_LIMIT and 0 < current <= peak <= DATA_LIMIT,
                'GPU_DEVICE_ALLOCATION_BOUND','Measured device allocation missing or exceeds its separate64MiB ceiling')
        return peak
    observed_peak=measured_device_allocation(allocations[0])
    n = len(submitted)
    require([integer(row,'submission') for row in submitted] == list(range(1,n+1))
            and [integer(row,'submission') for row in completed] == list(range(1,n+1))
            and [integer(row,'completed_commands') for row in completed] == list(range(1,n+1)),
            'GPU_TRACE_INCOMPLETE','Every submission must have one matching terminal completion')
    require(len(tiles) == n and all(row.get('terminal_success') == '1' and row.get('actual_gpu') == '1' for row in completed),
            'GPU_TRACE_INCOMPLETE','Native tile completion evidence is incomplete')
    groups = {}
    for index, row in enumerate(tiles,1):
        require(row.get('actual_gpu') == '1' and integer(row,'submissions') == index
                and integer(row,'completed_commands') == index, 'GPU_TRACE_INCOMPLETE','Tile order or completion counter differs')
        m,k,tokens = (integer(row,key) for key in ('M','K','N'))
        first,count = (integer(row,key) for key in ('tile_first','tile_rows'))
        require(k in (2048,4096) and 8 <= m <= 8192 and m%8 == 0 and tokens in (8,16,32,64,128)
                and count == min(8192,m-first) and first+count <= m, 'GPU_TRACE_SHAPE','Unadmitted GPU tile shape')
        require(all(submitted[index-1].get(key) == row.get(key) for key in ('weight','tile_first','tile_rows','M','K','N',
                    'borrowed_weight_bytes','borrowed_view_bytes','weight_view_offset','alias_identity_verified')),
                'GPU_TRACE_INCOMPLETE','Submitted command differs from completed tile')
        admission=submitted[index-1]
        memory={'pressure_level':integer(admission,'pressure'),'available_bytes':integer(admission,'available_bytes'),
            'rss_bytes':integer(admission,'rss_bytes'),'swap_growth_bytes':integer(admission,'swap_growth_bytes')}
        require(sufficient_memory_sample(memory,baseline_swap),'GPU_RESOURCE_ADMISSION','GPU submission lacks the sufficient native memory headroom')
        require(integer(row,'copy_weight_bytes') == 0 and integer(row,'borrowed_weight_bytes') == count*(k//32)*34
                and integer(row,'borrowed_weight_bytes') <= WEIGHT_TILE_BYTES and integer(row,'alias_identity_verified') == 1
                and integer(row,'weight_view_offset') < 65536 and integer(row,'weight_view_offset')%4 == 0
                and integer(row,'borrowed_weight_bytes')+integer(row,'weight_view_offset') <= integer(row,'borrowed_view_bytes') <= WEIGHT_TILE_BYTES+2*65536,
                'GPU_WEIGHT_COPY_BOUND','Exact bounded original-pointer NoCopy alias and zero copied weights required')
        require(integer(row,'allocation_count') == 3 and integer(row,'gpu_payload_current') == GPU_BYTES
                and integer(row,'tile_buffer_reuses') == index-1,
                'GPU_ALLOCATION_BOUND','Allocation count, retained payload or reuse counter differs')
        finite_number(row,'gpu_seconds'); finite_number(row,'command_and_readback_us'); finite_number(row,'copy_weight_us')
        attempt = integer(row,'attempt')
        require(attempt >= 1 and integer(row,'operation_buffer_reuses') == attempt-1,
                'GPU_TRACE_COUNTER','Operation reuse counter differs')
        group = groups.setdefault(attempt,{'shape':(m,k,tokens),'cursor':0,'tiles':0,'last_submission':0,
                    'view_bytes':integer(row,'borrowed_view_bytes')})
        require(group['shape'] == (m,k,tokens) and first == group['cursor'], 'GPU_TRACE_COVERAGE','Missing/overlapping GPU row tile')
        group['cursor'] += count; group['tiles'] += 1; group['last_submission'] = index
    seen = set()
    for operation_index,row in enumerate(published,1):
        attempt = integer(row,'attempt')
        require(attempt in groups and attempt not in seen, 'GPU_TRACE_PUBLICATION','Unknown or duplicate final publication')
        group = groups[attempt];m,k,tokens=group['shape']
        require(group['cursor'] == m and group['tiles'] == (m+8191)//8192
                and (integer(row,'M'),integer(row,'K'),integer(row,'N')) == (m,k,tokens),
                'GPU_TRACE_COVERAGE','Final publication does not cover complete output')
        cpu,gpu,total=(integer(row,key) for key in ('cpu_graph_workspace','gpu_workspace','total_workspace'))
        quant = tokens*(k//32)*34
        expected_provider = ((64+quant+63)&~63) + m*tokens*4
        view=integer(row,'borrowed_view_bytes')
        require(view == group['view_bytes'] and 0 < view <= WEIGHT_TILE_BYTES+2*65536 and gpu == GPU_BYTES and cpu+gpu+view == total and total <= DATA_LIMIT
                and integer(row,'provider_workspace') == expected_provider <= cpu, 'GPU_ALLOCATION_BOUND','Actual graph/Metal workspace exceeds budget')
        require(integer(row,'operations') == operation_index and integer(row,'submissions') == group['last_submission']
                and integer(row,'completed_commands') == group['last_submission'],
                'GPU_TRACE_COUNTER','Final operation or terminal command counter differs')
        require(row.get('publish') == '1' and row.get('actual_gpu') == '1','GPU_TRACE_PUBLICATION','Final native GPU publication is not explicit')
        peak=measured_device_allocation(row)
        require(peak >= observed_peak,'GPU_DEVICE_ALLOCATION_BOUND','Observed allocation peak decreases')
        observed_peak=peak
        seen.add(attempt)
    require(seen == set(groups), 'GPU_TRACE_PUBLICATION','GPU operation has no final publication')
    return {'passed':True,'actual_backend':BACKEND,'actual_gpu':True,'submitted_commands':n,
            'multi_token_shapes':sorted({int(row['N']) for row in tiles}),
            'zero_weight_copies':True,'all_weight_aliases_verified':True,
            'completed_commands':len(completed),'published_operations':len(published),
            'fixed_gpu_bytes':GPU_BYTES,'device_allocated_observed_peak_bytes':observed_peak,
            'device_allocation_limit_bytes':DATA_LIMIT,'unmeasured_driver_residency_bytes':None,'all_gpu_operations_terminal':True,
            'scope':'native bridge trace only; no model fidelity/EOS/global performance qualification'}
