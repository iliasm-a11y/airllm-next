"""Sequential reproduction. Reads only an existing local Ling checkpoint."""
import argparse
from pathlib import Path
import subprocess
import sys
from common import ROOT,REPO


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--torch-python',type=Path,help='Optional existing Python with CPU PyTorch; no installation/download')
    a=ap.parse_args()
    py=sys.executable
    commands=[
        [py,'research/exact_io/audit_metadata.py'],
        [py,'research/exact_io/routing.py'],
        [py,'research/exact_io/cache_reference.py'],
        [py,'research/exact_io/benchmark_compression.py','--model-dir',str(a.model_dir)],
        [py,'research/exact_io/exact_rank.py','--model-dir',str(a.model_dir)],
        [py,'research/exact_io/trained_parity.py','--model-dir',str(a.model_dir)],
        [py,'research/exact_io/cross_expert.py'],
        [py,'research/exact_io/scenarios.py'],
        ['clang','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',
            'research/exact_io/metal_probe.m','-o','research/exact_io/build/metal_probe'],
        [py,'research/exact_io/capture_environment.py'],
        [py,'research/exact_io/benchmark_transport.py'],
        [py,'-m','pytest','research/exact_io/tests','-q','--junitxml=research/exact_io/results/tests.xml'],
        [py,'-m','pytest','--noconftest','tests_next/test_speculative.py','-k','not native','-q',
            '--junitxml=research/exact_io/results/inherited-controls.xml']]
    if a.torch_python:
        commands.append([py,'research/exact_io/backend_probes.py','--python',str(a.torch_python)])
        commands.append([str(a.torch_python),'research/exact_io/numerical_contract.py',
                         '--expert-pack',str(ROOT/'build/trained-expert-pack')])
    import os
    env=dict(os.environ,PYTHONPATH=str(REPO/'air_llm'))
    with (ROOT/'results/reproduction.log').open('w') as log:
        for cmd in commands:
            print('RUN',cmd,flush=True)
            result=subprocess.run(cmd,cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            print(result.stdout,flush=True);log.write(str(cmd)+'\n'+result.stdout);log.flush()
            if result.returncode:raise SystemExit(result.returncode)
    subprocess.run([py,'research/exact_io/finalize_results.py'],cwd=REPO,env=env,check=True)


if __name__=='__main__':main()
