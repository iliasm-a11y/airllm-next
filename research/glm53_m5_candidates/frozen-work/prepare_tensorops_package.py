"""Stage a distinct Metal4 variant. Never alters the measured scalar provider."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

W=Path(__file__).resolve().parent
old=W/'metal-cpu-q8-bridge';new=W/'metal-tensorops-q8-candidate'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert not (new/'source').exists()
for name in ('source','driver','fixtures-warm-v2'):
    shutil.copytree(old/name,new/name,ignore=shutil.ignore_patterns('__pycache__','run1','build-receipt.json','source-manifest.json'))
for name in ('build_package.py','enable_bridge_objcxx.cmake','native_gpu_run.py','gpu_contract.py','profile.py','test_gpu_contract.py'):
    shutil.copy2(old/name,new/name)
source=(new/'q8_tensorops.metal').read_text()
header=new/'source/ggml/src/ggml-cpu/metal-q8-bridge-kernel.h'
header.write_text('#pragma once\n// Distinct unqualified Metal4 TensorOps candidate.\n#define AIRLLM_CPU_METAL_Q8_KERNEL_SHA256 "'+sha(new/'q8_tensorops.metal')+'"\nstatic const char * const airllm_cpu_metal_q8_source = R"Q8BRIDGE('+source+')Q8BRIDGE";\n')
mm=new/'source/ggml/src/ggml-cpu/metal-q8-bridge.mm'
value=mm.read_text()
for a,b in [
('if (!device) GGML_ABORT("AIRLLM_CPU_METAL_Q8 no Metal device");',
 'if (!device || ![device supportsFamily:MTLGPUFamilyApple10]) GGML_ABORT("AIRLLM_CPU_METAL_Q8 TensorOps requires Apple10 GPU");'),
('options.mathMode = MTLMathModeSafe;', 'options.mathMode = MTLMathModeSafe;\n        options.languageVersion = MTLLanguageVersion4_0;'),
('pipeline.maxTotalThreadsPerThreadgroup < 64','pipeline.maxTotalThreadsPerThreadgroup < 128'),
('MTLSizeMake((count+7)/8, (tokens+7)/8, 1) threadsPerThreadgroup:MTLSizeMake(8,8,1)',
 'MTLSizeMake((count+31)/32, (tokens+7)/8, 1) threadsPerThreadgroup:MTLSizeMake(128,1,1)')]:
    assert value.count(a)==1,a
    value=value.replace(a,b)
mm.write_text(value)
build=new/'build_package.py';value=build.read_text()
value=value.replace("str(WORK/'airllm-next/research/glm53_batched_prefill/metal_q8.metal')","str(HERE/'q8_tensorops.metal')")
value=value.replace('A metadata smoke may load Metal libraries but submits no GPU command.',
    'Metadata loads legacy Metal initialization (including empty commands), but no bridge matrix work.')
build.write_text(value)
profile=new/'profile.py';value=profile.read_text().replace("'kernel': 'validated q8_batch_dot component'","'kernel': 'experimental Metal4 MPP int8xint8 int32 dots32 ordered FP32 FMA'")
profile.write_text(value)
base=W/'metal-integration-v4-source/source'
manifest=json.loads((old/'source-manifest.json').read_text());overlay={};parts=[]
for relative in manifest['files']:
    candidate=new/'source'/relative;previous=base/relative
    original=previous.read_text() if previous.exists() else ''
    parts.extend(difflib.unified_diff(original.splitlines(True),candidate.read_text().splitlines(True),
        fromfile='a/'+relative if previous.exists() else '/dev/null',tofile='b/'+relative))
    overlay[relative]={'base_sha256':sha(previous) if previous.exists() else None,
                       'candidate_sha256':sha(candidate),'bytes':candidate.stat().st_size}
patch=new/'cpu-metal-q8-bridge.patch';patch.write_text(''.join(parts))
record={'schema_version':2,'status':'distinct source candidate; not built or executed',
    'build_default':'OFF','runtime_default':'OFF','patch_sha256':sha(patch),'files':overlay,
    'derivation':{'measured_scalar_provider_patch_sha256':manifest['patch_sha256'],
        'measured_scalar_summary_sha256':sha(W/'gpu-warm-fixture-summary.json'),
        'reason':'normal warm scalar-provider cases3.77..38.92x slower; test exact integer TensorOps before long GLM'},
    'base_reconstruction_receipt_sha256':sha(W/'metal-integration-v4-source/receipt.json'),
    'unchanged_driver_sha256':{name:sha(new/name) for name in manifest['driver_sha256']},
    'auxiliary_sha256':{name:sha(new/name) for name in ('build_package.py','profile.py','gpu_contract.py','native_gpu_run.py','enable_bridge_objcxx.cmake','q8_tensorops.metal')},
    'full_model_qualified':False,'numerical_qualified':False,'performance_qualified':False,
    'limits':'same original native guards, same fixed4buffers, same64MiB data/device and128MiB component cap',
    'scope':'new shader/dispatch/language/profile only; original provider and receipts unchanged'}
(new/'source-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'directory':str(new),'patch_sha256':sha(patch),'kernel_sha256':sha(new/'q8_tensorops.metal'),'original_unchanged':sha(old/'cpu-metal-q8-bridge.patch')==manifest['patch_sha256']}))
