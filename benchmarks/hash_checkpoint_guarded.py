#!/usr/bin/env python3
"""Compute a complete checkpoint SHA-256 under the unchanged resource guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    a.model=a.model.expanduser().resolve(strict=True)
    a.output=a.output.expanduser().absolute()
    sys.path.insert(0,str(ROOT/'air_llm'))
    from airllm.nextgen.experiment_guard import ExperimentLimits,supervise,validate_output_paths
    destinations=validate_output_paths(a.output,protected=[a.model,Path(__file__)])
    if any(path.exists() or path.is_symlink() for path in ([a.output] if a.worker else destinations)):
        raise FileExistsError('Refusing to overwrite checkpoint or prior evidence')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.worker:
        before=a.model.stat(); h=hashlib.sha256()
        with a.model.open('rb',buffering=0) as f:
            while block:=f.read(1<<20):h.update(block)
        after=a.model.stat()
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        stable=identity(before)==identity(after)
        with a.output.open('x') as f:
            json.dump(dict(sha256=h.hexdigest(),bytes=before.st_size,stable_file=stable),f,indent=2)
            f.write('\n')
        if not stable:raise RuntimeError('Checkpoint changed during hashing')
    else:
        supervise([sys.executable,str(Path(__file__).resolve()),'--model',str(a.model),
            '--output',str(a.output),'--worker'],a.output,
            ExperimentLimits(admitted_payload_bytes=64<<20,timeout_seconds=120),env=dict(os.environ))


if __name__=='__main__':main()
