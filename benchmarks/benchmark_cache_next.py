#!/usr/bin/env python3
"""Cache policy comparison on explicit synthetic routing distributions."""
import json
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'air_llm'))
import numpy as np
from airllm.nextgen.cache import WeightCache


def run():
    rng=np.random.default_rng(521);E=256;N=20000
    uniform=rng.integers(E,size=N);p=1/(np.arange(E)+1)**1.2;p=p/p.sum();zipf=rng.choice(E,N,p=p)
    scan=np.concatenate([rng.choice(12,800),np.arange(E)]*20)
    shifted=np.concatenate([rng.integers(s,s+16,N//4) for s in (0,64,128,192)])
    rows=[];value=np.zeros(4096,np.uint8)
    for name,trace in [('uniform',uniform),('zipf',zipf),('hot_plus_scan',scan),('shifting_hotset',shifted)]:
        for capacity in (8,32,64):
            for policy in ('lru','slru','lfu'):
                c=WeightCache(capacity*value.nbytes,policy);start=perf_counter()
                for i in trace:
                    key=int(i)
                    if c.get(key) is None:c.put(key,value)
                elapsed=perf_counter()-start;hits=c.stats['hits'];misses=c.stats['misses']
                rows.append({'distribution':name,'policy':policy,'entries_capacity':capacity,
                  'requests':len(trace),'hit_ratio':hits/(hits+misses),'miss_read_bytes':misses*value.nbytes,
                  'seconds':elapsed,'peak_bytes':c.peak_bytes,'budget_bytes':c.capacity})
    return {'scope':'synthetic traces; not real GLM route measurements','results':rows}


if __name__=='__main__':
    dest=Path(sys.argv[1] if len(sys.argv)>1 else 'evidence-next/cache-policy.json')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(run(),indent=2));print(dest)
