"""Bind existing successful build receipts and actual files; never compile/run.

This command refuses incomplete/unchanged CPU-only candidates. It cannot create
a pending binding or fill missing build success/metadata fields on behalf of a
builder. Raw build receipts and capture receipts are never changed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as p


def library_records(directory):
    paths = [path for path in Path(directory).resolve().glob('*.dylib') if not path.is_symlink() and path.is_file()]
    if not paths:
        raise ValueError('actual native library files required')
    return {path.name: p.descriptor(path) for path in paths}


def arm(build_path, libraries, runtime_manifest, helper):
    build = p.read(build_path)
    command = build.get('command', [])
    if command.count('-o') != 1:
        raise ValueError('exact successful helper compile output is required')
    binary = Path(command[command.index('-o') + 1])
    libs = library_records(libraries)
    return {'build_receipt': p.descriptor(build_path), 'binary': p.descriptor(binary),
            'helper_source': p.descriptor(helper), 'runtime_manifest': p.descriptor(runtime_manifest),
            'libraries': libs, 'runtime_binaries': {'helper': p.sha(binary), **{name: row['sha256'] for name, row in libs.items()}}}


def create(args):
    prepared = p.read(p.HERE / 'prepare.json')
    modules = p.load_modules(prepared)
    original = p.read(p.TAIL / 'prepare-manifest.json')
    reference_build = Path(original['original_build_receipt'])
    reference_command = p.read(reference_build)['command']
    original_libraries = Path(reference_command[reference_command.index('-L') + 1])
    a = arm(reference_build, original_libraries, args.reference_runtime_manifest, original['original_helper_source'])
    b = arm(args.candidate_build, args.candidate_libraries, args.candidate_runtime_manifest,
            p.GPU / 'driver/research/glm53_batched_prefill/native_prefill.cpp')
    a['source_sha256'] = original['reference_source_identity']
    helper = Path(b['helper_source']['path'])
    source_paths = {'helper_source': helper, 'snapshot_source': helper.with_name('snapshot.h'),
        'state_capture_source': helper.parent.parent / 'glm53_gpu_v3/state_capture.inc',
        'native_manifest': args.candidate_runtime_manifest, 'wrapper_source': p.GPU / 'native_gpu_run.py',
        'gpu_contract_source': p.GPU / 'gpu_contract.py', 'gpu_profile_source': p.GPU / 'profile.py',
        'gpu_profile_document': args.gpu_profile, 'provider_build_receipt': args.provider_build,
        'gpu_driver_build_receipt': args.candidate_build,
        'native_guard_source': p.REPO / 'air_llm/airllm/nextgen/glm_native.py'}
    b['source_sha256'] = {name: p.sha(path) for name, path in source_paths.items()}
    binding = {'schema_version': 1, 'status': 'actual_two_build_binding',
        'prepared_sha256': p.sha(p.HERE / 'prepare.json'), 'reference': a, 'candidate': b,
        'provider_build_receipt': p.descriptor(args.provider_build), 'gpu_profile': p.descriptor(args.gpu_profile),
        'candidate_libraries_directory': str(args.candidate_libraries.resolve()),
        'helper_patch': p.descriptor(p.HERE / 'helper-cpu-to-gpu.patch'),
        'full_model_qualified': False, 'snapshot_exact_qualified': False,
        'scope': 'existing real build/source bytes only; no native process or numerical validation performed'}
    # Validate the entire contract before exposing a final binding. The
    # temporary is our own new metadata file, never a changed capture receipt.
    temporary = args.output.with_suffix(args.output.suffix + '.unvalidated.tmp')
    if args.output.exists() or temporary.exists():
        raise FileExistsError(args.output)
    try:
        temporary.write_text(json.dumps(binding, indent=2, allow_nan=False) + '\n')
        p.load_pair(temporary, prepared, modules)
        p.verify_sources(prepared)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-runtime-manifest', 'candidate-build', 'provider-build', 'candidate-runtime-manifest',
                 'candidate-libraries', 'gpu-profile', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    binding = create(args)
    print(json.dumps({'status': binding['status'], 'binding_sha256': p.sha(args.output),
                      'candidate_binary_sha256': binding['candidate']['binary']['sha256'],
                      'model_loaded': False, 'compiler_calls': 0, 'gpu_commands': 0}))


if __name__ == '__main__':
    main()
