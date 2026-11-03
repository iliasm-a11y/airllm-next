"""Run only under guarded_build_once; actual pinned native FP16 conversion."""
from pathlib import Path
import hashlib
import json
import subprocess
HERE=Path(__file__).resolve().parent;W=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=W/'metal-integration-v4-source/source'
library=Path('/Users/macbook/Documents/Codex/2026-11-02/referenced-chatgpt-conversation-this-is-an-2/work/runtime-v4/bin')
baseline=json.loads((W/'native-prefill-final-controls.build.json').read_text())
binary=HERE/'resource-query';output=HERE/'resource-query-receipt.json'
assert not binary.exists() and not output.exists()
assert sha(library/'libggml-base.0.28.0.dylib')==baseline['native_libraries_sha256']['libggml-base.0.28.0.dylib']
tracked={str(p):sha(p) for p in (Path(__file__),HERE/'resource_query.cpp',source/'ggml/include/ggml.h',library/'libggml-base.0.28.0.dylib')}
command=['/Library/Developer/CommandLineTools/usr/bin/clang++','-isysroot','/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk',
    '-x','objective-c++','-std=c++17','-O3','-framework','Foundation','-framework','IOKit','-I',str(source/'ggml/include'),str(HERE/'resource_query.cpp'),'-x','none','-L',str(library),'-lggml-base',
    '-Wl,-rpath,'+str(library),'-o',str(binary)]
record={'scope':'exhaustive half-scale predicate equivalence plus resident CPU scale scan; not full GPU/GLM performance',
        'input_sha256':tracked,'command':command,'model_loaded':False,'gpu_commands':0}
try:
    built=subprocess.run(command,capture_output=True,text=True,timeout=120)
    record['compile']={'returncode':built.returncode,'stdout':built.stdout,'stderr':built.stderr}
    assert built.returncode==0
    result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
    record.update(binary_sha256=sha(binary),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    assert result.returncode==0
    record['measurement']=json.loads(result.stdout);record['status']='completed_predicate_audit_only'
except Exception as error:record.update(status='failed',error=str(error))
finally:
    record['input_sha256_after']={p:sha(Path(p)) for p in tracked}
    assert record['input_sha256_after']==tracked
    output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'output':str(output)}))
raise SystemExit(0 if record['status']=='completed_predicate_audit_only' else 2)
