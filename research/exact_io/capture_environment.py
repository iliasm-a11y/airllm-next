"""Capture task-relevant environment facts without serials, hostnames or secrets."""
import importlib.util
import argparse
import json
import platform
import shutil
import subprocess
import sys
from common import ROOT, REPO, save


def run(output='environment.json'):
    result=subprocess.run(['system_profiler','SPHardwareDataType','-json'],capture_output=True,text=True)
    try:
        hw=json.loads(result.stdout)['SPHardwareDataType'][0]
        hw={k:v for k,v in hw.items() if k in ['chip_type','machine_model','machine_name','physical_memory','number_processors']}
    except (ValueError,KeyError,IndexError):hw={'unavailable':True}
    probe=ROOT/'build/metal_probe'
    try:
        p=subprocess.run([str(probe)],capture_output=True,text=True)
        metal=json.loads(p.stdout)
    except (OSError,ValueError):metal={'probe_executable_unavailable':True}
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True).stdout.strip()
    dirty=subprocess.run(['git','status','--porcelain'],cwd=REPO,capture_output=True,text=True).stdout
    save(output,dict(platform=platform.platform(),python=platform.python_version(),
        hardware=hw,metal=metal,python_executable=sys.executable,
        modules={m:importlib.util.find_spec(m) is not None for m in
            ['numpy','torch','mlx','safetensors','pytest']},
        base_commit='d43032f38a0325aafd2973847d7ed84b21280c61',source_commit=git,
        source_tree_dirty=bool(dirty),
        network_cli={'probed_this_capture':False},
        note='Environment inventory only; module presence and device discovery do not establish a measured model runtime.'))
    print(hw,metal,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='environment.json',help='JSON filename under results/')
    run(ap.parse_args().output)
