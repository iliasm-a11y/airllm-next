"""Run under guarded_build_once. No model or Metal initialization."""
from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent;W=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=W/'metal-integration-v4-source/source'
library=Path('/Users/macbook/Documents/Codex/2026-11-02/referenced-chatgpt-conversation-this-is-an-2/work/runtime-v4/bin')
baseline=json.loads((W/'native-prefill-final-controls.build.json').read_text())
binary=HERE/'component';output=HERE/'receipt.json'
assert not binary.exists() and not output.exists()
for name,digest in baseline['native_libraries_sha256'].items():assert sha(library/name)==digest
tracked={str(p):sha(p) for p in (Path(__file__),HERE/'component.cpp',source/'ggml/include/ggml.h',source/'ggml/include/ggml-backend.h',source/'ggml/include/ggml-cpu.h',*[library/n for n in baseline['native_libraries_sha256']])}
command=['/Library/Developer/CommandLineTools/usr/bin/clang++','-isysroot','/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk',
    '-std=c++17','-O3','-I',str(source/'ggml/include'),'-I',str(source/'vendor'),str(HERE/'component.cpp'),'-L',str(library),
    '-lggml','-lggml-base','-lggml-cpu','-Wl,-rpath,'+str(library),'-o',str(binary)]
record={'scope':'native CPU10 resident synthetic actual-shaped matmul/RMS comparison N16/32/64/128 vs split8; no model causality or speed proof',
    'inputs_sha256':tracked,'command':command,'model_loaded':False,'gpu_matrix_dispatches':0}
try:
    built=subprocess.run(command,capture_output=True,text=True,timeout=120)
    record['compile']={'returncode':built.returncode,'stdout':built.stdout,'stderr':built.stderr}
    assert built.returncode==0
    result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=90)
    record.update(binary_sha256=sha(binary),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    assert result.returncode==0
    record['measurement']=json.loads(result.stdout);record['status']='completed_component_diagnostic_only'
except Exception as error:record.update(status='failed',error=repr(error))
finally:
    record['inputs_sha256_after']={p:sha(Path(p)) for p in tracked}
    assert record['inputs_sha256_after']==tracked
    output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'output':str(output)}))
raise SystemExit(0 if record['status']=='completed_component_diagnostic_only' else 2)
