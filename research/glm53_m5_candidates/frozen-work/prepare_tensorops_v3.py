"""Freeze the actually pipeline-compiled shader into a new isolated package."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
W=Path(__file__).resolve().parent
old=W/'metal-tensorops-q8-candidate';new=W/'metal-tensorops-q8-v3'
assert not new.exists();new.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('source','driver','fixtures-warm-v2'):
    shutil.copytree(old/name,new/name,ignore=shutil.ignore_patterns('__pycache__','run1','build-receipt.json','source-manifest.json'))
for name in ('build_package.py','enable_bridge_objcxx.cmake','native_gpu_run.py','gpu_contract.py','profile.py','test_gpu_contract.py'):
    shutil.copy2(old/name,new/name)
shutil.copy2(W/'metal-tensorops-q8-variants/v3-int8-types.metal',new/'q8_tensorops.metal')
source=(new/'q8_tensorops.metal').read_text()
(new/'source/ggml/src/ggml-cpu/metal-q8-bridge-kernel.h').write_text(
    '#pragma once\n// Pipeline-compiled source; numerical/speed qualification remains separate.\n#define AIRLLM_CPU_METAL_Q8_KERNEL_SHA256 "'+sha(new/'q8_tensorops.metal')+'"\nstatic const char * const airllm_cpu_metal_q8_source = R"Q8BRIDGE('+source+')Q8BRIDGE";\n')
base=W/'metal-integration-v4-source/source'
original=json.loads((old/'source-manifest.json').read_text());overlay={};parts=[]
for name in original['files']:
    a=base/name;b=new/'source'/name
    parts.extend(difflib.unified_diff(a.read_text().splitlines(True) if a.exists() else [],b.read_text().splitlines(True),
        fromfile='a/'+name if a.exists() else '/dev/null',tofile='b/'+name))
    overlay[name]={'base_sha256':sha(a) if a.exists() else None,'candidate_sha256':sha(b),'bytes':b.stat().st_size}
patch=new/'cpu-metal-q8-bridge.patch';patch.write_text(''.join(parts))
record={'schema_version':2,'status':'distinct v3 shader compiled only, provider not built or executed',
    'build_default':'OFF','runtime_default':'OFF','files':overlay,'patch_sha256':sha(patch),
    'shader_compile_only_receipt_sha256':sha(W/'metal-tensorops-q8-variants/v3-compile.json'),
    'derivation':{'original_provider_patch_sha256':json.loads((W/'metal-cpu-q8-bridge/source-manifest.json').read_text())['patch_sha256'],
        'first_tensorops_patch_sha256':original['patch_sha256'],'kernel_api_corrections':['slice arguments must be lvalues','explicit int8_t operand type instead of distinct char']},
    'auxiliary_sha256':{name:sha(new/name) for name in ('build_package.py','profile.py','gpu_contract.py','native_gpu_run.py','enable_bridge_objcxx.cmake','q8_tensorops.metal')},
    'full_model_qualified':False,'numerical_qualified':False,'performance_qualified':False,
    'scope':'same guards4buffers and quotas; new actual shader/patch/package identities required'}
(new/'source-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
(new/'README.md').write_text('''# Metal4 TensorOps v3 — isolated experiment

Source compiled into a Metal4 pipeline in a separate compile-only process.
The first two source versions were refused (temporary slice arguments, then
plain char instead of explicit int8_t). Those failures and exact sources remain
preserved; neither is a hardware numerical failure. This source uses explicit
int8xint8→int32 TensorOps per32-term block and ordered FP32 FMA per scale block.
No weights/quantizer, native guards, buffers or limits are changed.

Host compile, component equality/speed and full GLM qualification remain
separate. Build/runtime OFF by default. No positive result is inherited.
The component suite starts with smaller resident matrices, uses five graph
iterations, and keeps the previous128MiB RSS ceiling. Large M8192/K4096/N64,128
remain excluded after the previous scalar-provider resource refusal.
''')
print(json.dumps({'directory':str(new),'patch_sha256':sha(patch),'kernel_sha256':sha(new/'q8_tensorops.metal')}))
