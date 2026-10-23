#!/usr/bin/env python3
"""Hash an existing GGUF in a supervised process, without mmap or copying it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'air_llm'))
from airllm.nextgen.experiment_guard import ExperimentLimits, supervise, validate_output_paths
from airllm.nextgen.physical_io import physical_read_bytes, set_fd_nocache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    if len(args.expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in args.expected_sha256):
        p.error('Expected lowercase SHA256 hex digest')
    validate_output_paths(args.output,protected=[args.model])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not args.worker:
        supervise([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],args.output,
                  ExperimentLimits(rss_bytes=128<<20,admitted_payload_bytes=64<<20,timeout_seconds=120))
        return
    start=perf_counter();disk=physical_read_bytes();digest=hashlib.sha256();count=0
    fd=os.open(args.model,os.O_RDONLY)
    try:
        if sys.platform=='darwin':set_fd_nocache(fd)
        while True:
            chunk=os.read(fd,4<<20)
            if not chunk:break
            digest.update(chunk);count+=len(chunk)
        stat=os.fstat(fd)
    finally:os.close(fd)
    end_disk=physical_read_bytes()
    result={'file_bytes':stat.st_size,'read_bytes':count,'sha256':digest.hexdigest(),
            'expected_sha256':args.expected_sha256,'match':digest.hexdigest()==args.expected_sha256,
            'seconds':perf_counter()-start,'buffer_bytes':4<<20,'mmap':False,
            'fd_nocache':sys.platform=='darwin','process_storage_read_bytes':
                end_disk-disk if disk is not None and end_disk is not None else None,
            'scope':'Integrity hash and process storage counter; not a NAND bandwidth certificate.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if not result['match']:raise ValueError('Checkpoint SHA256 mismatch')
    print(json.dumps(result))


if __name__=='__main__':main()
