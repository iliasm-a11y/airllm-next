"""Isolated bounded-conversion memory benchmark, never an inference benchmark.

Invoke this same script with baseline and candidate source paths in PYTHONPATH.
Each subprocess must start fresh so process peak RSS is comparable. Input tiles
are generated deterministically; no complete floating point matrix is retained.
"""
import argparse
import hashlib
import json
import platform
import resource
import sys
import tempfile
from pathlib import Path
from time import perf_counter
import numpy as np
from airllm.nextgen.sidecar import QuantizedSidecar


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rows',type=int,default=32768)
    parser.add_argument('--columns',type=int,default=4096)
    parser.add_argument('--tile-rows',type=int,default=64)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(197)
    tiles_digest=hashlib.sha256()
    def tiles():
        for start in range(0,args.rows,args.tile_rows):
            tile=rng.standard_normal((min(args.tile_rows,args.rows-start),args.columns),dtype=np.float32)
            tiles_digest.update(memoryview(tile).cast('B'))
            yield tile
    with tempfile.TemporaryDirectory(prefix='airllm-conversion-') as directory:
        sidecar=QuantizedSidecar(directory,'deterministic-fixture',create=True)
        begin=perf_counter()
        sidecar.add('fixture.experts.0.weight',tiles(),(args.rows,args.columns))
        elapsed=perf_counter()-begin
        entry=sidecar.entries['fixture.experts.0.weight']
        packed=sidecar.load('fixture.experts.0.weight',rows=(0,1))
        row_digest=hashlib.sha256()
        for value in (packed.weight,packed.scales,packed.biases):row_digest.update(memoryview(np.ascontiguousarray(value)).cast('B'))
        report={'scope':'CPU Linux conversion microbenchmark; no model inference',
            'platform':platform.platform(),'python':sys.version,'source_module':sys.modules[QuantizedSidecar.__module__].__file__,
            'shape':[args.rows,args.columns],'tile_rows':args.tile_rows,'seed':197,
            'input_tile_sha256':tiles_digest.hexdigest(),'first_packed_row_sha256':row_digest.hexdigest(),
            'seconds':elapsed,'packed_payload_bytes':entry['nbytes'],
            'relative_l2_error':entry['relative_l2_error'],'max_abs_error':entry['max_abs_error'],
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'rss_scope':'fresh process lifetime; includes interpreter and deterministic tile generation',
            'cache_state':'temporary output; OS page cache uncontrolled','physical_ssd_throughput_certified':False}
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
