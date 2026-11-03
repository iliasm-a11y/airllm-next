"""Start the unchanged authorized uncached functional pilot in a clean child context.

No model/guard/campaign source is patched. Native/null reasoning remains Max in
the frozen GLM template. This is functional research, not numeric qualification.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys

W=Path(__file__).resolve().parent
campaign=W/'airllm-next/research/glm53_batched_prefill/campaign.py'
config=W/'cpu10-reference.json'
output=W/'agents-cpu16-native'
cwd=W/'agents-cpu16-empty-cwd'
receipt=W/'agents-cpu16-launch.json'
if output.exists() or cwd.exists() or receipt.exists():
    raise FileExistsError('Fresh functional output, launch receipt and empty working directory required')
prefixes=('AIRLLM_','GGML_','LLAMA_','DYLD_')
removed={prefix:sum(name.startswith(prefix) for name in os.environ) for prefix in prefixes}
environment={name:value for name,value in os.environ.items() if not name.startswith(prefixes)}
command=[sys.executable,str(campaign),'--config',str(config),'--output',str(output),
    '--mode','agents','--microbatch','16','--logical-batch','16','--gpu-layers','0',
    '--tokens','512','--reasoning-effort','native','--agent-order','1,600,1',
    '--request-seconds','2700','--max-seconds','7200']
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
paths=[Path(__file__),campaign,config,
    W/'airllm-next/research/glm53_shared_prefill/qualify_agents.py',
    W/'airllm-next/air_llm/airllm/nextgen/glm_native.py']
record={'scope':'launch context only; no numeric/EOS/functional qualification',
    'started_at_utc':datetime.now(timezone.utc).isoformat(),'command':command,
    'input_sha256':{str(path):sha(path) for path in paths},
    'working_directory':str(cwd),'working_directory_initially_empty':True,
    'child_environment_removed_prefix_counts':removed,
    'source_files_modified':False,'shared_cache':False,'production_changed':False,
    'reasoning_effort':'native/null; preserves frozen native Max',
    'ambient_presets_scope':'explicit variable prefixes removed and fresh empty cwd; actual argv/props still must be verified',
    'all_qualification_fields':False}
cwd.mkdir(mode=0o700)
receipt.write_text(json.dumps(record,indent=2)+'\n')
receipt.chmod(0o600)
os.chdir(cwd)
os.execvpe(command[0],command,environment)
