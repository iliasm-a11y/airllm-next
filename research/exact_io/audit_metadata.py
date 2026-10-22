"""Recompute counts from inherited headers; no weight download and no inference."""
import collections
import json
import math
import re
from common import ROOT, digest, save

SIZES = {'F32':4,'F16':2,'BF16':2,'F8_E4M3':1,'F8_E4M3FN':1}


def audit(name, file, layers, experts, k, moe_layers):
    path = ROOT/'inherited'/file
    headers = json.loads(path.read_text())
    totals = collections.Counter()
    groups = collections.defaultdict(collections.Counter)
    rows = []
    for shard, h in headers.items():
        for key, (dtype, shape, offsets) in h['tensors'].items():
            count = math.prod(shape)
            n = offsets[1]-offsets[0]
            assert n == count*SIZES[dtype], (key, n)
            m = re.search(r'(?:model\.)?layers\.(\d+)\.', key)
            mtp = key.startswith('mtp.') or (m is not None and int(m.group(1)) >= layers)
            scale = 'scale' in key
            routed = '.mlp.experts.' in key
            embedding = 'embed_tokens.weight' in key
            part = 'mtp' if mtp else 'routed' if routed else 'embedding' if embedding else 'always'
            groups[part]['stored_bytes'] += n
            groups[part]['scales' if scale else 'non_scale_elements'] += count
            totals['bytes'] += n
            totals['tensors'] += 1
            if not scale:
                totals['non_scale_elements'] += count
            rows.append((key, part, dtype, shape, n))
    main = sum(g['non_scale_elements'] for part,g in groups.items() if part != 'mtp')
    totals['header_bytes'] = sum(8+h['header_len'] for h in headers.values())
    totals['files_bytes_including_headers'] = totals['bytes']+totals['header_bytes']
    routed_bytes = groups['routed']['stored_bytes']
    # Exact per-expert storage equality is checked, not inferred from average size.
    by_layer = collections.Counter()
    for key, part, dtype, shape, n in rows:
        if part == 'routed':
            by_layer[int(re.search(r'layers\.(\d+)\.', key).group(1))] += n
    assert len(by_layer) == moe_layers
    assert all(n % experts == 0 for n in by_layer.values())
    active_routed_bytes = sum(n//experts*k for n in by_layer.values())
    emb_rows = [r for r in rows if r[1] == 'embedding']
    one_embedding_row = sum(n//shape[0] for _,_,_,shape,n in emb_rows)
    out = dict(name=name, evidence='recomputed from inherited full safetensors headers',
        source_header_sha256=digest(path), shards=len(headers), totals=dict(totals),
        components={a:dict(b) for a,b in groups.items()},
        main_non_scale_elements=main,
        checkpoint_non_scale_elements=totals['non_scale_elements'],
        active_routed_stored_bytes_per_position=active_routed_bytes,
        always_stored_bytes_per_pass=groups['always']['stored_bytes'],
        selective_embedding_row_bytes=one_embedding_row,
        naive_decode_stored_bytes=groups['always']['stored_bytes']+active_routed_bytes+one_embedding_row,
        weights_only_residency_10GB_best_case_bytes=max(0,groups['always']['stored_bytes']-10e9)+active_routed_bytes+one_embedding_row,
        storage_2TB_ratio_needed=2e12/totals['bytes'],
        note='Non-scale elements include buffers. Counts are not nn.Parameter counts. No claim about runtime FP8 kernel equivalence.')
    return out


if __name__ == '__main__':
    rows = [audit('GLM-5.3 FP8','glm53_headers.json',78,256,8,75),
            audit('Qwen3.8 BF16','qwen38_bf16_headers.json',92,512,10,92)]
    save('metadata.json',rows)
    for r in rows:
        print(r['name'],r['checkpoint_non_scale_elements'],r['totals']['bytes'],
              'decode_GB',r['naive_decode_stored_bytes']/1e9,flush=True)
