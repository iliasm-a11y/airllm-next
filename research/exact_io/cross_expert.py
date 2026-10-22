"""Test a reversible inter-expert XOR transform, accounting for the anchor."""
from collections import defaultdict
from compression import zstd
import json
from common import ROOT,save
from airllm.nextgen.storage import OffsetCheckpoint
import numpy as np


def run():
    provenance=json.loads((ROOT/'results/sample_provenance.json').read_text())['samples']
    cp=OffsetCheckpoint(ROOT/'samples')
    groups=defaultdict(list)
    for i,p in enumerate(provenance):
        key=p['key'];parts=key.split('.')
        group=(int(parts[2]),parts[-2],p['part'])
        groups[group].append((int(parts[5]),f'sample.{i}.weight'))
    rows=[]
    for g,items in groups.items():
        raw=[cp.read_array(k).tobytes() for _,k in sorted(items)]
        if len(raw)!=3:raise ValueError('Expected three aligned experts')
        anchor=np.frombuffer(raw[0],np.uint8)
        deltas=[np.bitwise_xor(anchor,np.frombuffer(r,np.uint8)).tobytes() for r in raw[1:]]
        for original,d in zip(raw[1:],deltas):
            assert np.bitwise_xor(anchor,np.frombuffer(d,np.uint8)).tobytes()==original
        independent=sum(len(zstd.compress(r,level=1)) for r in raw)
        xor=len(zstd.compress(raw[0],level=1))+sum(len(zstd.compress(d,level=1)) for d in deltas)
        rows.append(dict(layer=g[0],projection=g[1],part=g[2],raw_bytes=sum(map(len,raw)),
            independent_zstd_bytes=independent,anchor_plus_xor_zstd_bytes=xor,
            xor_ratio_vs_independent=xor/independent,all_bits_equal=True))
    cp.close()
    summary=dict(raw_bytes=sum(r['raw_bytes'] for r in rows),
                 independent_bytes=sum(r['independent_zstd_bytes'] for r in rows),
                 xor_bytes=sum(r['anchor_plus_xor_zstd_bytes'] for r in rows))
    summary['xor_ratio_vs_independent']=summary['xor_bytes']/summary['independent_bytes']
    save('cross_expert.json',dict(evidence='measured compression of aligned real Ling weight slices',results=rows,summary=summary,
        limitation='One reversible XOR transform, three experts per layer. Does not rule out a better conditional code.'))
    print(summary,flush=True)


if __name__=='__main__':run()
