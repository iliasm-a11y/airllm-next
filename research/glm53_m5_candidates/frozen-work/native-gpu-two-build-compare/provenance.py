"""Fail-closed source/build/key contract for the isolated GPU comparison.

The binding is generated after real builds. Pending/null candidate fingerprints
are never accepted. Reading this module loads no native runtime or model.
"""
from __future__ import annotations
import copy
import difflib
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK / 'airllm-next'
GPU = WORK / 'metal-cpu-q8-bridge'
TAIL = WORK / 'native-tail-even16-candidate'
POLICY = 'cpu8-tail-compatible-v1'
REFERENCE_BINARY = '62a7d2578c58e96839be2d4c79d8c414c8d535e5a1adac54820043f9e695f4b0'
REFERENCE_SOURCE = '558aa441ed74aca6b295613ccb414c4b1f2bf2134a6bdc0ae32a09b35c32c395'
REFERENCE_BUILD = '4cf35cfba893c14a04f8ff8eb05876faced75fede08c8f24a6af0b069b4ceb93'
TAIL_PARAMETERS = {'prefill_chunk_policy': POLICY, 'prefill_chunk_policy_version': 1, 'tail_reference_microbatch': 8}
DELEGATES = {
    'native': ('research.glm53_batched_prefill.native_compare', REPO / 'research/glm53_batched_prefill/native_compare.py'),
    'batch': ('research.glm53_batched_prefill.batch_compare', REPO / 'research/glm53_batched_prefill/batch_compare.py'),
    'fidelity': ('research.glm53_m5.compare_fidelity', REPO / 'research/glm53_m5/compare_fidelity.py'),
    'key': ('research.glm53_batched_prefill.snapshot_key', REPO / 'research/glm53_batched_prefill/snapshot_key.py'),
    'guard': ('air_llm.airllm.nextgen.glm_native', REPO / 'air_llm/airllm/nextgen/glm_native.py'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 8 << 20:
        raise ValueError('bounded existing provenance JSON required: ' + str(path))
    return json.loads(path.read_text())


def hash_value(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('actual lowercase SHA256 required')
    return value


def descriptor(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def checked_file(record):
    if set(record) != {'path', 'sha256', 'bytes'} or type(record['bytes']) is not int or record['bytes'] < 1:
        raise ValueError('complete actual file descriptor required')
    hash_value(record['sha256'])
    path = Path(record['path']).resolve()
    if not path.is_file() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
        raise ValueError('actual file bytes differ from binding: ' + str(path))
    return path


def verify_sources(prepared):
    if prepared.get('status') != 'reviewed_source_not_built_or_model_qualified':
        raise ValueError('GPU source review/preparation incomplete; no actual build may be bound yet')
    pins = prepared['source_pins']
    required = {str(path.resolve()) for _, path in DELEGATES.values()}
    required |= {str((GPU / name).resolve()) for name in ('gpu_contract.py', 'profile.py', 'native_gpu_run.py')}
    required |= {str((HERE / name).resolve()) for name in ('provenance.py', 'numerics.py', 'gpu_compare.py', 'build_binding.py')}
    if not required <= set(pins):
        raise ValueError('complete strict delegated/source pin set required')
    for path, digest in pins.items():
        if sha(path) != hash_value(digest):
            raise ValueError('pinned comparison/runtime source changed: ' + path)
    for name, path in DELEGATES.values():
        cached = sys.modules.get(name)
        if cached is not None and Path(getattr(cached, '__file__', '')).resolve() != path.resolve():
            raise ValueError('strict delegate imported from another path: ' + name)


class CheckedModule:
    def __init__(self, module, prepared):
        self.module, self.prepared = module, prepared

    def __getattr__(self, name):
        value = getattr(self.module, name)
        if not callable(value):
            return value
        def call(*args, **kwargs):
            verify_sources(self.prepared)
            answer = value(*args, **kwargs)
            verify_sources(self.prepared)
            return answer
        return call


def load_modules(prepared):
    verify_sources(prepared)
    sys.path[:0] = [str(REPO), str(REPO / 'air_llm')]
    result = {key: CheckedModule(importlib.import_module(name), prepared)
              for key, (name, _) in DELEGATES.items()}
    spec = importlib.util.spec_from_file_location('bound_gpu_contract', GPU / 'gpu_contract.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result['gpu'] = CheckedModule(module, prepared)
    verify_sources(prepared)
    return result


def build_input(build, suffix):
    selected = [(Path(path), digest) for path, digest in build['inputs_sha256'].items() if path.endswith('/' + suffix)]
    if len(selected) != 1:
        raise ValueError('exact unique build input required: ' + suffix)
    path, digest = selected[0]
    if sha(path) != hash_value(digest):
        raise ValueError('actual build input changed: ' + str(path))
    return path.resolve(), digest


def verify_build(arm, *, reference=False):
    path = checked_file(arm['build_receipt'])
    build = read(path)
    binary = checked_file(arm['binary'])
    helper = checked_file(arm['helper_source'])
    if type(build.get('returncode')) is not int or build['returncode'] != 0:
        raise ValueError('real successful helper build required')
    if build.get('binary_sha256') != arm['binary']['sha256']:
        raise ValueError('helper build/binding binary differ')
    command = build.get('command')
    if not isinstance(command, list) or command.count('-o') != 1 or Path(command[command.index('-o') + 1]).resolve() != binary:
        raise ValueError('actual helper compile output differs')
    source_path, source_sha = build_input(build, 'glm53_batched_prefill/native_prefill.cpp')
    if source_path != helper or source_sha != arm['helper_source']['sha256']:
        raise ValueError('actual helper source does not match build')
    for suffix, key in (('glm53_batched_prefill/snapshot.h', 'snapshot_source'), ('glm53_gpu_v3/state_capture.inc', 'state_capture_source')):
        _, digest = build_input(build, suffix)
        if digest != arm['source_sha256'][key]:
            raise ValueError('built snapshot/capture dependency differs')
    if reference:
        if (arm['binary']['sha256'] != REFERENCE_BINARY or arm['helper_source']['sha256'] != REFERENCE_SOURCE or
                arm['build_receipt']['sha256'] != REFERENCE_BUILD):
            raise ValueError('original measured CPU8 build identity required')
    else:
        if build.get('prefill_chunk_policy', build.get('policy')) != POLICY:
            raise ValueError('GPU helper build must declare the CPU8-compatible tail policy')
        if (build.get('schema_version') != 2 or build.get('inputs_before') != build.get('inputs_sha256') or
                build.get('inputs_after') != build.get('inputs_sha256') or build.get('status') != 'built_metadata_only_smoked'):
            raise ValueError('GPU helper build lacks exact before/after source/header verification')
        for suffix in ('include/llama.h', 'ggml/include/ggml.h', 'ggml/include/ggml-backend.h', 'vendor/nlohmann/json.hpp'):
            build_input(build, suffix)
    binaries = {'helper': arm['binary']['sha256']}
    for name, record in arm['libraries'].items():
        path = checked_file(record)
        if path.name != name or path.is_symlink():
            raise ValueError('actual non-alias library name required')
        binaries[name] = record['sha256']
    if binaries != arm['runtime_binaries']:
        raise ValueError('actual library package differs from helper build or binding')
    recorded_libraries = build.get('native_libraries_sha256', {})
    if reference:
        if recorded_libraries != {k: v for k, v in binaries.items() if k != 'helper'}:
            raise ValueError('original library build fingerprints differ')
    else:
        directories = {Path(row['path']).parent for row in arm['libraries'].values()}
        if len(directories) != 1 or not recorded_libraries:
            raise ValueError('one actual candidate package directory required')
        directory = directories.pop()
        for logical, digest in recorded_libraries.items():
            if type(logical) is not str or '/' in logical or '\\' in logical:
                raise ValueError('logical candidate library name required')
            path = (directory / ('lib' + logical + '.dylib')).resolve()
            if path.name not in arm['libraries'] or str(path) != arm['libraries'][path.name]['path'] or digest != binaries.get(path.name):
                raise ValueError('logical build library differs from actual versioned file')
        runtime_manifest = read(checked_file(arm['runtime_manifest']))
        recorded = {Path(row['path']).name: row.get('sha256') for row in runtime_manifest['artifacts']
                    if row['path'].endswith('.dylib') and 'symlink' not in row}
        if recorded != {name: row['sha256'] for name, row in arm['libraries'].items()}:
            raise ValueError('actual complete versioned library package differs from runtime manifest')
    return build


def load_pair(binding_path, prepared, modules):
    verify_sources(prepared)
    binding = read(binding_path)
    if binding.get('schema_version') != 1 or binding.get('status') != 'actual_two_build_binding':
        raise ValueError('actual completed GPU build binding required; pending/null hashes are forbidden')
    if binding.get('prepared_sha256') != sha(HERE / 'prepare.json'):
        raise ValueError('binding belongs to another pinned preparation')
    a, b = binding['reference'], binding['candidate']
    builds = {'reference': verify_build(a, reference=True), 'candidate': verify_build(b)}
    for name in ('snapshot_source', 'state_capture_source'):
        if a['source_sha256'][name] != b['source_sha256'][name] or a['source_sha256'][name] != prepared['shared_source_sha256'][name]:
            raise ValueError('snapshot/state-capture source must remain identical')
    if a['source_sha256'] != prepared['reference_source_sha256'] or a['runtime_binaries'] != prepared['reference_runtime_binaries']:
        raise ValueError('reference source/runtime identity differs from measured CPU8')
    if b['helper_source']['sha256'] != prepared['candidate_helper_source_sha256']:
        raise ValueError('GPU helper differs from independently reviewed source')
    patch = checked_file(binding['helper_patch'])
    actual = ''.join(difflib.unified_diff(Path(a['helper_source']['path']).read_text().splitlines(True),
        Path(b['helper_source']['path']).read_text().splitlines(True),
        fromfile='a/research/glm53_batched_prefill/native_prefill.cpp', tofile='b/research/glm53_batched_prefill/native_prefill.cpp'))
    if actual != patch.read_text() or binding['helper_patch']['sha256'] != prepared['helper_patch_sha256']:
        raise ValueError('actual two helper sources differ from reviewed exact patch')
    provider_path = checked_file(binding['provider_build_receipt'])
    provider = read(provider_path)
    if provider.get('status') != 'built' or provider.get('CPU_METAL_Q8_BRIDGE_COMPILED') is not True:
        raise ValueError('real completed provider build required')
    if (provider.get('schema_version') != 2 or provider.get('inputs_before') != provider.get('inputs_sha256') or
            provider.get('inputs_after') != provider.get('inputs_sha256')):
        raise ValueError('provider exact before/after source verification missing')
    for path, digest in prepared['provider_source_pins'].items():
        if provider.get('inputs_sha256', {}).get(path) != digest or sha(path) != digest:
            raise ValueError('provider was not built from reviewed unchanged source: ' + path)
    profile_path = checked_file(binding['gpu_profile'])
    profile = modules['gpu'].validate_profile(read(profile_path))
    modules['gpu'].validate_artifacts(profile, provider, builds['candidate'],
        libraries=binding['candidate_libraries_directory'], binary=b['binary']['path'],
        helper_source=b['helper_source']['path'], runner_source=GPU / 'native_gpu_run.py',
        guard_source=REPO / 'air_llm/airllm/nextgen/glm_native.py', provider_receipt_sha256=sha(provider_path))
    if provider.get('provider_patch_sha256') != prepared['provider_patch_sha256']:
        raise ValueError('provider build patch differs from reviewed patch')
    for side, arm in (('reference', a), ('candidate', b)):
        helper = Path(arm['helper_source']['path'])
        source_paths = {'helper_source': helper, 'snapshot_source': helper.with_name('snapshot.h'),
            'state_capture_source': helper.parent.parent / 'glm53_gpu_v3/state_capture.inc',
            'native_manifest': checked_file(arm['runtime_manifest']),
            'wrapper_source': REPO / 'research/glm53_batched_prefill/native_run.py' if side == 'reference' else GPU / 'native_gpu_run.py'}
        if side == 'candidate':
            source_paths.update(gpu_contract_source=GPU / 'gpu_contract.py', gpu_profile_source=GPU / 'profile.py',
                gpu_profile_document=profile_path, provider_build_receipt=provider_path,
                gpu_driver_build_receipt=Path(arm['build_receipt']['path']),
                native_guard_source=REPO / 'air_llm/airllm/nextgen/glm_native.py')
        if arm['source_sha256'] != {name: sha(path) for name, path in source_paths.items()}:
            raise ValueError('actual helper/wrapper/build source bytes differ from binding: ' + side)
    verify_sources(prepared)
    return binding, builds, profile


def dependencies(report, receipt, arm, build, modules, *, gpu_profile=None):
    manifest = receipt['preflight']['snapshot_manifest']
    key, _ = modules['key'].canonical_key(manifest)
    if key != report.get('snapshot_key_sha256') or key != receipt['preflight'].get('snapshot_key_sha256'):
        raise ValueError('own native/report key was not recomputed from complete actual dependencies')
    if report.get('state_parameters') != manifest['state_parameters']:
        raise ValueError('report.state_parameters differs from its own keyed parameters')
    if receipt['preflight'].get('source_sha256') != arm['source_sha256']:
        raise ValueError('captured helper/wrapper/runtime source identity differs from bound build')
    combined = hashlib.sha256(json.dumps(arm['source_sha256'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if manifest['runtime_source_sha256'] != combined or manifest['runtime_binaries'] != arm['runtime_binaries'] or receipt['preflight'].get('runtime_binaries') != arm['runtime_binaries']:
        raise ValueError('captured runtime hashes differ from exact two builds')
    ub = report.get('microbatch')
    params = manifest['state_parameters']
    if ub not in (8, 16, 32, 64, 128) or params.get('n_ubatch') != ub or params.get('n_batch') != report.get('logical_batch'):
        raise ValueError('physical/logical batch metadata differs')
    normalized = copy.deepcopy(manifest)
    if gpu_profile is None:
        if ub != 8 or any(name in params for name in TAIL_PARAMETERS):
            raise ValueError('original CPU8 control required')
        if build.get('parameter_smoke', {}).get(str(ub)) != params:
            raise ValueError('CPU parameters differ from actual reference binary smoke')
        if (any(name.startswith('gpu_') for name in manifest['render_options']) or
                any(name.startswith('AIRLLM_CPU_METAL') for name in manifest['render_options']['native_environment'])):
            raise ValueError('CPU control unexpectedly contains GPU profile fields')
    else:
        modules['gpu'].validate_state(params, gpu_profile, ub)
        smoke = build.get('parameter_smoke', {}).get(str(ub), {})
        raw = json.loads(smoke.get('stdout', 'null'))
        if not isinstance(raw, dict):
            raise ValueError('actual GPU helper metadata smoke required')
        raw.pop('native_linkage', None)
        raw.pop('native_resource_context', None)
        if type(smoke.get('returncode')) is not int or smoke.get('returncode') != 0 or raw != params or smoke.get('state_parameters') != params:
            raise ValueError('GPU run parameters differ from actual binary build smoke')
        for name, value in TAIL_PARAMETERS.items():
            if type(params.get(name)) is not type(value) or params.get(name) != value:
                raise ValueError('CPU8-compatible tail metadata differs')
        if report.get('prefill_chunk_policy') != POLICY or report.get('prefill_chunk_policy_qualified') is not False:
            raise ValueError('GPU helper tail policy is absent or falsely qualified')
        # These exact GPU descriptor values were validated above. Unknown
        # parameter differences survive normalization and therefore reject.
        for name in TAIL_PARAMETERS:
            del normalized['state_parameters'][name]
        for name in ('execution_backend', 'cpu_metal_q8_bridge_compiled', 'gpu_execution_possible', 'gpu_bridge_requested',
                     'gpu_policy_sha256', 'weight_storage_policy', 'permanent_gpu_weight_bytes'):
            del normalized['state_parameters'][name]
        extension = gpu_profile['render_options_extension']
        for name, value in extension.items():
            if normalized['render_options'].get(name) != value:
                raise ValueError('keyed GPU execution profile differs')
            del normalized['render_options'][name]
        env = normalized['render_options']['native_environment']
        for name, value in gpu_profile['environment'].items():
            if env.get(name) != value:
                raise ValueError('keyed GPU environment differs')
            del env[name]
    for name in ('n_batch', 'n_ubatch'):
        del normalized['state_parameters'][name]
    # No receipt is changed: normalization is a local compatibility comparison,
    # after separate own-key and exact two-build validation.
    normalized['runtime_source_sha256'] = 'independently-validated-two-build-source-identity'
    normalized['runtime_binaries'] = 'independently-validated-two-build-binary-identity'
    return normalized


def check_chunks(report, batches, *, gpu=False):
    """Verify actual full-arm receipts against the bound source scheduling."""
    if report['mode'] != 'full':
        return  # Restore/split boundaries are checked by the unchanged strict comparator.
    ub = report['microbatch']
    for request, row in enumerate(report['requests']):
        expected = []
        offset = 0
        while offset < row['prompt_tokens']:
            raw = min(ub, row['prompt_tokens'] - offset)
            take = raw - raw % 8 if gpu and raw > 8 and raw % 2 else raw
            expected.append((offset, take, raw))
            offset += take
        selected = [batch for batch in batches if batch['request'] == request]
        if [(r['offset'], r['tokens']) for r in selected] != [(o, n) for o, n, _ in expected]:
            raise ValueError('actual chunks violate the bound CPU8/tail scheduling')
        if gpu:
            for actual, (_, take, raw) in zip(selected, expected):
                if (type(actual.get('unadjusted_take')) is not int or actual.get('unadjusted_take') != raw or
                        actual.get('odd_tail_split') is not (take != raw) or actual.get('prefill_chunk_policy') != POLICY):
                    raise ValueError('actual GPU tail-policy chunk receipt missing or false')
