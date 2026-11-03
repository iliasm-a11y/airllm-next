"""Explicit compile-only probe; called under the outer original build guards."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
HERE=Path(__file__).resolve().parent
binary=HERE/'compile-shader-only-run2'
receipt=HERE/'compile-only-receipt-run2.json'
assert not binary.exists() and not receipt.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=[Path(__file__),HERE/'compile_shader_only.mm',HERE/'q8_tensorops.metal']
before={str(p):sha(p) for p in sources}
record={'scope':'actual Metal4 shader/pipeline compilation only; no matrix/device command queue, numerical or performance qualification',
        'inputs_sha256':before,'matrix_dispatches':0,'model_loaded':False,'status':'building'}
command=['/Library/Developer/CommandLineTools/usr/bin/clang++','-isysroot','/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk','-std=c++17','-O2','-fobjc-arc',
         '-framework','Foundation','-framework','Metal',str(HERE/'compile_shader_only.mm'),'-o',str(binary)]
try:
    built=subprocess.run(command,capture_output=True,text=True,timeout=120)
    record['host_compile']={'command':command,'returncode':built.returncode,'stdout':built.stdout,'stderr':built.stderr}
    if built.returncode:raise RuntimeError('Host compiler refused')
    record['binary_sha256']=sha(binary)
    result=subprocess.run([str(binary),str(HERE/'q8_tensorops.metal')],capture_output=True,text=True,timeout=30)
    record['shader_compile']={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    if result.returncode:raise RuntimeError('Metal shader/pipeline compiler refused')
    record['actual_descriptor']=json.loads(result.stdout)
    record['status']='shader_and_pipeline_compiled_only'
except Exception as error:record.update(status='failed',error=str(error))
finally:
    record['inputs_after']={str(p):sha(p) for p in sources}
    if record['inputs_after']!=before:record.update(status='failed',error='Compile input identity changed')
    receipt.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'receipt':str(receipt)}))
raise SystemExit(0 if record['status']=='shader_and_pipeline_compiled_only' else 2)
