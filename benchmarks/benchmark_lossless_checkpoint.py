"""Compare bounded original/standalone readers on one complete local checkpoint.

This measures warm local file reads, compression, and exact byte transport.
It does not measure generation or physical external SSD throughput. The original
checkpoint is only read; the complete temporary pack is deleted on exit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import tempfile
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'air_llm'))
from airllm.nextgen.lossless import LosslessCheckpoint,pack_checkpoint
from airllm.nextgen.storage import OffsetCheckpoint,encoded_read_bytes,physical_read_bytes
from airllm.ultra.checkpoint import atomic_json


def requests(checkpoint,tile_bytes):
    for key,entry in sorted(checkpoint.tensors.items()):
        shape=entry['shape']
        if len(shape) in (2,3):
            experts=range(shape[0]) if len(shape)==3 else (None,)
            rows=shape[-2]
            row_bytes=entry['bytes']//max(1,rows*(shape[0] if len(shape)==3 else 1))
            if row_bytes>tile_bytes:raise MemoryError(f'One row of {key} exceeds the read tile budget')
            step=max(1,tile_bytes//max(1,row_bytes))
            for expert in experts:
                for start in range(0,rows,step):
                    yield {'key':key,'expert':expert,'rows':(start,min(rows,start+step))}
        else:
            if entry['bytes']>tile_bytes:raise MemoryError(f'Untiled {key} exceeds the read tile budget')
            yield {'key':key}


def pass_once(reader,selections,*,digest=False):
    result=hashlib.sha256() if digest else None
    for request in selections:
        value=reader.read_array(**request)
        if result is not None:result.update(memoryview(value).cast('B'))
    return result.hexdigest() if result is not None else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model')
    p.add_argument('--output',required=True)
    p.add_argument('--max-total-bytes',type=int,default=256*1024**2)
    p.add_argument('--block-bytes',type=int,default=1024**2)
    p.add_argument('--tile-bytes',type=int,default=4*1024**2)
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--codec',choices=['auto','raw','zlib','shuffle-zlib','palette16'],default='auto')
    a=p.parse_args()
    if a.repeats<1 or a.tile_bytes<1:p.error('repeats and tile-bytes must be positive')
    with OffsetCheckpoint(a.model,max_read_bytes=max(a.tile_bytes,a.block_bytes)) as original:
        if original.total_bytes>a.max_total_bytes:
            raise MemoryError('Explicit pack byte allowance is smaller than selected tensors')
        selections=list(requests(original,a.tile_bytes))
        with tempfile.TemporaryDirectory(prefix='airllm-standalone-') as work:
            destination=Path(work)/'pack';start=perf_counter()
            meta=pack_checkpoint(original,destination,standalone=True,codec=a.codec,
                block_bytes=a.block_bytes,max_total_bytes=a.max_total_bytes)
            conversion_seconds=perf_counter()-start
            weight_files={entry['file'] for entry in original.tensors.values()}
            original_weight_bytes=sum((original.root/name).stat().st_size for name in weight_files)
            standalone_total_bytes=sum(path.stat().st_size for path in destination.rglob('*') if path.is_file())
            with LosslessCheckpoint(destination,max_read_bytes=a.tile_bytes,
                                    max_block_bytes=a.block_bytes) as packed:
                digest=pass_once(original,selections,digest=True)
                assert pass_once(packed,selections,digest=True)==digest
                variants={'original':original,'standalone':packed}
                samples={name:[] for name in variants};rng=random.Random(814)
                for _ in range(a.repeats):
                    order=list(variants);rng.shuffle(order)
                    for name in order:
                        reader=variants[name]
                        encoded=encoded_read_bytes(reader);logical=reader.stats['logical_bytes']
                        physical=physical_read_bytes();start=perf_counter()
                        pass_once(reader,selections)
                        seconds=perf_counter()-start;end_physical=physical_read_bytes()
                        samples[name].append({'seconds':seconds,
                            'requested_tensor_bytes':reader.stats['logical_bytes']-logical,
                            'encoded_file_bytes':encoded_read_bytes(reader)-encoded,
                            'process_physical_read_bytes':None if physical is None or end_physical is None else end_physical-physical})
                medians={name:statistics.median(row['seconds'] for row in values) for name,values in samples.items()}
                report={'scope':'Complete local checkpoint, warm byte transport; no generation or external SSD claim',
                    'platform':platform.platform(),'python':sys.version,'codec':a.codec,'block_bytes':a.block_bytes,
                    'tile_bytes':a.tile_bytes,'hash_verification':True,'source_free_reader':True,
                    'tensor_count':len(original.tensors),'selections_per_pass':len(selections),
                    'source_tensor_payload_bytes':meta['raw_bytes'],'source_weight_file_bytes':original_weight_bytes,
                    'standalone_payload_bytes':meta['stored_bytes'],'standalone_total_bytes':standalone_total_bytes,
                    'standalone_payload_ratio':meta['stored_bytes']/max(1,meta['raw_bytes']),
                    'conversion_seconds':conversion_seconds,'transport_sha256':digest,
                    'median_seconds':medians,'standalone_speed_ratio':medians['original']/medians['standalone'],
                    'samples':samples,'process_lifetime_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                    'rss_scope':'Entire benchmark process including conversion; not a per-reader memory comparison',
                    'cache_state':'Both readers warmed by integrity pass; OS page cache uncontrolled'}
                atomic_json(a.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
