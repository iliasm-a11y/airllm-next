"""Replay unit-sized expert requests through the real default AirLLM cache.

This is a reference-policy experiment, not allocation/SSD profiling. Native
attention/head weights are deliberately absent from this simplified trace.
"""
import json
import numpy as np
from common import ROOT,save
from airllm.nextgen.cache import WeightCache


def run():
    z=np.load(ROOT/'inherited/routing_traces_olmoe.npz')
    ids=z['ids_long'];L,C,_,_=ids.shape;N=int(z['N'])
    value=np.zeros(1,np.uint8)
    out=[]
    for capacity in [32,64,128,256]:
        misses=hits=requests=0
        for c in range(C):
            test=ids[:,c,512:1024,:].astype(np.int64)+np.arange(L)[:,None,None]*N
            seq=np.transpose(np.sort(test,axis=-1),(1,0,2)).reshape(-1)
            cache=WeightCache(capacity,policy='slru')
            for item in seq:
                key=int(item)
                if cache.get(key,1) is None:cache.put(key,value)
            misses+=cache.stats['misses'];hits+=cache.stats['hits'];requests+=len(seq)
            assert cache.bytes<=capacity
        out.append(dict(capacity_expert_units=capacity,misses=misses,hits=hits,requests=requests))
        print(out[-1],flush=True)
    save('cache_reference.json',dict(evidence='actual AirLLM WeightCache SLRU replay, unit-sized abstraction',
        results=out,excludes_attention_head_and_decoded_expansion=True,physical_io_measured=False))


if __name__=='__main__':run()
