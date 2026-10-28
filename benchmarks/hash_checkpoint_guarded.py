#!/usr/bin/env python3
"""Compute a complete checkpoint SHA-256 under the unchanged resource guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
HASH_READ_BYTES=1<<20


def file_identity(stat):
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def checkpoint_hash(model):
    """Hash every byte; Darwin avoids populating the OS data cache for this FD.

    F_NOCACHE changes only the separately opened hash descriptor. It neither
    purges existing cached pages nor changes model-forward reads. A failed
    request is fatal: an ordinary-cache fallback would silently change policy.
    """
    before=model.stat()
    digest=hashlib.sha256()
    policy='darwin_f_nocache' if sys.platform=='darwin' else 'ordinary_os_cache'
    with model.open('rb',buffering=0) as stream:
        opened=os.fstat(stream.fileno())
        if file_identity(before)!=file_identity(opened):
            raise RuntimeError('Checkpoint changed before hash descriptor opened')
        if sys.platform=='darwin':
            sys.path.insert(0,str(ROOT/'air_llm'))
            from airllm.nextgen.physical_io import set_fd_nocache
            set_fd_nocache(stream.fileno())
        count=0
        while block:=stream.read(HASH_READ_BYTES):
            digest.update(block)
            count+=len(block)
        closed=os.fstat(stream.fileno())
    after=model.stat()
    stable=(file_identity(before)==file_identity(closed)==file_identity(after)
            and count==before.st_size)
    return dict(sha256=digest.hexdigest(),bytes=count,stable_file=stable,
                hash_io_policy=policy,hash_io_policy_applied=True,
                hash_read_buffer_bytes=HASH_READ_BYTES,
                hash_io_scope='checkpoint_hash_descriptor_only')


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
        result=checkpoint_hash(a.model)
        with a.output.open('x') as f:
            json.dump(result,f,indent=2)
            f.write('\n')
        if not result['stable_file']:raise RuntimeError('Checkpoint changed during hashing')
    else:
        supervise([sys.executable,str(Path(__file__).resolve()),'--model',str(a.model),
            '--output',str(a.output),'--worker'],a.output,
            ExperimentLimits(admitted_payload_bytes=64<<20,timeout_seconds=120),env=dict(os.environ))


if __name__=='__main__':main()
