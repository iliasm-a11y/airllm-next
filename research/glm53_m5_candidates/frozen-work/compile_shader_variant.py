"""Compile a fresh Metal source with an actual pinned compile-only native tool."""
import argparse
from pathlib import Path
import hashlib
import json
import subprocess
p=argparse.ArgumentParser();p.add_argument('--kernel',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
W=Path(__file__).resolve().parent
prior=json.loads((W/'metal-tensorops-q8-candidate/compile-only-receipt-run2.json').read_text())
binary=W/'metal-tensorops-q8-candidate/compile-shader-only-run2'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(binary)==prior['binary_sha256']
tracked={str(path.resolve()):sha(path) for path in (Path(__file__),a.kernel,binary,W/'metal-tensorops-q8-candidate/compile_shader_only.mm')}
result=subprocess.run([str(binary),str(a.kernel.resolve())],capture_output=True,text=True,timeout=30)
record={'scope':'shader/pipeline compilation only; existing original guarded build supervisor owns UID/power/resources',
        'input_sha256':tracked,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
        'status':'shader_and_pipeline_compiled_only' if result.returncode==0 else 'shader_compile_failed',
        'matrix_dispatches':0,'model_loaded':False,
        'input_sha256_after':{path:sha(Path(path)) for path in tracked}}
if result.returncode==0:record['actual_descriptor']=json.loads(result.stdout)
assert record['input_sha256']==record['input_sha256_after']
a.output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'output':str(a.output)}))
raise SystemExit(0 if result.returncode==0 else 2)
