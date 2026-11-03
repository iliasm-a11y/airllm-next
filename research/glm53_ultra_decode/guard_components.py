"""Supervise small synthetic components with the existing owned-process guard."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from airllm.nextgen.experiment_guard import ExperimentLimits,supervise

def run(args):
    output=args.output.resolve()
    supervision=output.with_name(output.name+'-supervision')
    guard_file=supervision.with_suffix('.guard.json')
    if output.exists() or guard_file.exists():raise FileExistsError(output)
    command=[sys.executable,'-m','research.glm53_ultra_decode.components',
        '--runtime',str(args.runtime.resolve()),'--build',str(args.build.resolve()),
        '--output',str(output),'--format',str(args.format),'--rows',str(args.rows),
        '--width',str(args.width),'--banks',str(args.banks),'--seed',str(args.seed),
        '--repetitions',str(args.repetitions)]
    env={k:v for k,v in os.environ.items() if not k.startswith(('DYLD_','AIRLLM_','GGML_','LLAMA_ARG_'))}
    limits=ExperimentLimits(rss_bytes=int(8.5*(1<<30)),reserve_bytes=3<<30,
        swap_growth_bytes=32<<20,admitted_payload_bytes=256<<20,
        timeout_seconds=180,interval_seconds=.25,max_processes=8)
    failure=None
    try:
        report=supervise(command,supervision,limits=limits,env=env)
    except (RuntimeError,MemoryError) as error:
        failure=str(error)
        report=json.loads(guard_file.read_text()) if guard_file.exists() else {}
    result=json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else {}
    guarded=not failure and report.get('exit_code')==0 and report.get('violation') is None and not report.get('cleanup_error')
    accepted=guarded and result.get('component_fidelity_passed') is True
    print(json.dumps({'guard_status':'completed' if guarded else 'failed','component_fidelity_passed':accepted,
                      'guard_violation':report.get('violation'),'error':result.get('error',failure)}))
    return 0 if accepted else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('runtime','build','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--format',type=int,choices=(2,8),required=True)
    p.add_argument('--rows',type=int,default=1536);p.add_argument('--width',type=int,default=4096)
    p.add_argument('--banks',type=int,choices=(1,2),default=1)
    p.add_argument('--seed',type=int,default=5303101);p.add_argument('--repetitions',type=int,default=15)
    raise SystemExit(run(p.parse_args()))
