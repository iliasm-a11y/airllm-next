"""Compile the small CPU/BLAS scheduler fixture only. Never execute it."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
SOURCE=WORK/'metal-integration-v4-source/source'
LIB=Path('/Users/macbook/Documents/Codex/2026-11-02/referenced-chatgpt-conversation-this-is-an-2/work/runtime-v4/bin')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
baseline=json.loads((WORK/'native-prefill-final-controls.build.json').read_text())
binary=HERE/'component';receipt=HERE/'build-receipt.json'
if binary.exists() or receipt.exists():raise FileExistsError('Fresh binary/receipt required')
for name,expected in baseline['native_libraries_sha256'].items():
 if sha(LIB/name)!=expected:raise ValueError('Pinned library differs: '+name)
paths=[Path(__file__),HERE/'component.cpp',*[SOURCE/'ggml/include'/name for name in ('ggml.h','ggml-backend.h','ggml-cpu.h','ggml-blas.h')],SOURCE/'vendor/nlohmann/json.hpp',*[LIB/name for name in baseline['native_libraries_sha256']]]
tracked={str(p):sha(p) for p in paths}
command=['/Library/Developer/CommandLineTools/usr/bin/clang++','-isysroot','/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk','-std=c++17','-O3','-Wno-deprecated-declarations','-I',str(SOURCE/'ggml/include'),'-I',str(SOURCE/'vendor'),str(HERE/'component.cpp'),'-L',str(LIB),'-Wl,-rpath,'+str(LIB),'-lggml','-lggml-base','-lggml-cpu','-lggml-blas','-o',str(binary)]
record={'status':'building','scope':'build only; native fixture/model/GPU not executed','started_at_utc':datetime.now(timezone.utc).isoformat(),'command':command,'inputs_sha256':tracked,'component_executed':False,'model_loaded':False,'gpu_matrix_dispatches':0}
try:
 env={k:v for k,v in os.environ.items() if not k.startswith(('AIRLLM_','GGML_','DYLD_','LLAMA_ARG_')) and k!='LLAMA_API_KEY'}
 start=time.perf_counter();result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
 record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,compile_seconds=time.perf_counter()-start)
 if result.returncode:raise ValueError('Fixture compiler failed')
 record.update(status='built_not_executed',binary_sha256=sha(binary),native_libraries_sha256=baseline['native_libraries_sha256'])
except Exception as error:record.update(status='failed',error=repr(error))
finally:
 record['inputs_sha256_after']={p:sha(p) for p in tracked};record['inputs_unchanged']=record['inputs_sha256_after']==tracked
 if not record['inputs_unchanged']:record['status']='failed'
 receipt.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'receipt':str(receipt),'component_executed':False}))
raise SystemExit(0 if record['status']=='built_not_executed' else 2)
