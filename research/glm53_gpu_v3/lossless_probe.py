"""Bounded exact packed-byte codec diagnostic; no new model representation installed."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,statistics,time
from compression import zstd
from .system_memory import SystemMemoryObserver
from .process_io import ProcessIOObserver

MIB=1024**2

def codec_sample(raw):
    if not isinstance(raw,bytes) or not 0<len(raw)<=8*MIB:raise ValueError('sample must be1byte..8MiB')
    t=time.perf_counter();packed=zstd.compress(raw,level=3);prepare=time.perf_counter()-t
    if len(packed)>len(raw)+MIB:raise MemoryError('compressed payload unexpectedly exceeds bound')
    times=[]
    for repeat in range(8):
        t=time.perf_counter();decoder=zstd.ZstdDecompressor();decoded=decoder.decompress(packed,max_length=len(raw)+1);elapsed=time.perf_counter()-t
        if not decoder.eof or decoder.unused_data or decoded!=raw:raise ValueError('lossless byte equality failed')
        if repeat:times.append(elapsed)
        del decoded,decoder
    return {'original_bytes':len(raw),'compressed_bytes':len(packed),'ratio':len(packed)/len(raw),
            'sha256_original':hashlib.sha256(raw).hexdigest(),'exact':True,'compress_seconds':prepare,
            'decode_seconds':times,'decode_median_seconds':statistics.median(times),
            'original_bytes_per_decode_second':len(raw)/statistics.median(times)}

def run(path,plan,output):
    if output.exists():raise ValueError('refusing to overwrite evidence')
    rows=json.loads(Path(plan).read_text());rows=rows.get('ranges') if isinstance(rows,dict) else rows
    if not rows or sum(r['length'] for r in rows)>64*MIB or any(type(r['offset'])is not int or r['offset']<0 or type(r['length'])is not int or not 0<r['length']<=8*MIB for r in rows):raise ValueError('invalid <=64MiB sample plan')
    st=path.stat()
    if any(r['offset']+r['length']>st.st_size for r in rows):raise ValueError('sample beyond original file')
    sys=SystemMemoryObserver();proc=ProcessIOObserver(os.getpid());initial=sys.snapshot()
    if not initial.get('available') or initial['pressure_level']!=1:raise RuntimeError('normal measured pressure required')
    result={'schema':1,'status':'running','source_file':path.name,'file_stat':{'size':st.st_size,'mtime_ns':st.st_mtime_ns},'samples':[],
            'system_before':initial,'process_before':proc.snapshot(),'scope':'level3 Zstandard packed-byte samples; warm component decode only, no full-model or SSD speed claim'}
    try:
        with Path('/tmp/airllm-glm53-benchmark.lock').open('a') as lock,path.open('rb',buffering=0) as source:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for row in rows:
                source.seek(row['offset']);raw=source.read(row['length'])
                if len(raw)!=row['length']:raise OSError('short sample read')
                result['samples'].append({**row,**codec_sample(raw)});del raw
                mem=sys.snapshot();p=proc.snapshot()
                if not mem.get('available') or mem['pressure_level']==4 or mem['swap_used_bytes']-initial['swap_used_bytes']>32*MIB:raise RuntimeError('system memory guard rejected codec pilot')
                if not p.get('available') or p['rusage']['resident_bytes']>256*MIB:raise RuntimeError('component RSS/measurement guard')
            final=path.stat()
            if (st.st_size,st.st_mtime_ns,st.st_ino)!=(final.st_size,final.st_mtime_ns,final.st_ino):raise RuntimeError('source stats changed')
            result.update(status='complete',model_file_stats_unchanged=True,system_after=sys.snapshot(),process_after=proc.snapshot())
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        output.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gguf',type=Path,required=True);p.add_argument('--ranges',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run(a.gguf,a.ranges,a.output);print(json.dumps({'status':r['status'],'samples':len(r['samples'])}))
