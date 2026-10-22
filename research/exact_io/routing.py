"""Replay true inherited OLMoE routes. These are teacher-forced, not free decode.

Cache simulations and multi-position union counts measure request reuse only.
They do not measure SSD bytes, future acceptance or target-model speed.
"""
from collections import OrderedDict, Counter, defaultdict, deque
import heapq
import numpy as np
from common import ROOT, digest, save


def union_counts(blocks):
    s = np.sort(blocks, axis=-1)
    return 1 + np.sum(s[..., 1:] != s[..., :-1], axis=-1)


def cache_misses(requests, capacity, policy):
    cache = OrderedDict()
    freq = Counter()
    misses = 0
    for key in requests:
        key = int(key)
        freq[key] += 1
        if key in cache:
            cache.move_to_end(key)
            continue
        misses += 1
        if not capacity:
            continue
        if len(cache) == capacity:
            if policy == 'lru':
                cache.popitem(last=False)
            else:
                victim = min(cache, key=lambda x: freq[x])
                if freq[key] < freq[victim]:
                    continue
                del cache[victim]
        cache[key] = None
    return misses


def bypass_opt(requests, capacity):
    """Offline Belady with optional bypass; lower bound for this unit-item trace."""
    n = len(requests)
    future = {}
    nexts = np.full(n, n+1, np.int64)
    for i in range(n-1,-1,-1):
        key = int(requests[i])
        nexts[i] = future.get(key, n+1)
        future[key] = i
    cache = {}
    heap = []
    misses = 0
    for i,key in enumerate(requests):
        key, nxt = int(key), int(nexts[i])
        if key not in cache:
            misses += 1
            if not capacity:
                continue
            if len(cache) == capacity:
                while heap and cache.get(heap[0][1]) != -heap[0][0]:
                    heapq.heappop(heap)
                far, victim = heap[0]
                if nxt >= -far:
                    continue
                heapq.heappop(heap)
                del cache[victim]
        cache[key] = nxt
        heapq.heappush(heap, (-nxt,key))
    return misses


def run():
    path = ROOT/'inherited/routing_traces_olmoe.npz'
    z = np.load(path)
    ids = z['ids_long']
    L,C,T,k = ids.shape
    N = int(z['N'])
    assert N == 64 and k == 8
    cats = z['cats_long'].tolist()
    assert np.all(np.diff(np.sort(ids,axis=-1),axis=-1) > 0)
    rng = np.random.default_rng(20261022)
    starts = rng.integers(0,T-65,size=256)
    # Same start points at every depth make marginal union differences comparable.
    curves = []
    for n in range(1,66):
        block = ids[:,:,starts[:,None]+np.arange(n),:].reshape(L,C,256,n*k)
        u = union_counts(block)
        curves.append(u.mean(axis=(0,2)).tolist())
    window_rows = []
    for n in [1,2,4,8,16,32,64,128,256,512]:
        nt = T//n*n
        blocks = ids[:,:,:nt].reshape(L,C,nt//n,n*k)
        u = union_counts(blocks)
        window_rows.append(dict(n=n,mean_experts_per_layer=float(u.mean()),
            grouped_read_reduction=n*k/float(u.mean()),
            uniform_iid_expected=N*(1-(1-k/N)**n),
            per_category={c:float(u[:,j].mean()) for j,c in enumerate(cats)}))
    # Calibrate static pins on an earlier disjoint part of each sequence.
    # Evaluate the same 512-token range for all policies; expert key includes layer.
    caches = []
    percat = []
    for c in range(C):
        train = ids[:,c,:512,:].astype(np.int64)+np.arange(L)[:,None,None]*N
        test = ids[:,c,512:1024,:].astype(np.int64)+np.arange(L)[:,None,None]*N
        requests = np.transpose(np.sort(test,axis=-1),(1,0,2)).reshape(-1)
        ranking = [v for v,_ in Counter(train.reshape(-1).tolist()).most_common()]
        for capacity in [0,32,64,128,256]:
            pinned = set(ranking[:capacity])
            static = sum(int(x) not in pinned for x in requests)
            row = dict(category=cats[c],capacity_expert_units=capacity,requests=len(requests),
                cold_static_misses=static+capacity,
                static_misses_after_pin_setup=static,
                lru_misses=cache_misses(requests,capacity,'lru'),
                lfu_admit_misses=cache_misses(requests,capacity,'lfu'),
                offline_bypass_opt_misses=bypass_opt(requests,capacity))
            percat.append(row)
    for cap in [0,32,64,128,256]:
        selected = [r for r in percat if r['capacity_expert_units']==cap]
        caches.append({'capacity_expert_units':cap, **{k:sum(r[k] for r in selected) for k in
            ['requests','cold_static_misses','static_misses_after_pin_setup','lru_misses','lfu_admit_misses','offline_bypass_opt_misses']}})
    out = dict(evidence='reanalysis of inherited trained OLMoE teacher-forced traces',
        trace_sha256=digest(path),shape=list(ids.shape),N=N,k=k,
        no_new_model_forward=True,free_generation_acceptance_measured=False,
        source_validation='Inherited two-layer FP32 comparison only; not reproduced here',
        union_curves_by_category=curves,windows=window_rows,cache_summary=caches,cache_by_category=percat,
        caveats=['Routes on draft mistakes can differ from these true-prefix teacher-forced routes.',
        'Equal-size expert units exclude attention/head/decoded weight expansion.',
        'Static pin setup is included once in cold_static_misses; calibration is extra work.',
        'Belady observes the entire future and is not an implementable online predictor.',
        'No OLMoE-to-GLM/Qwen route transfer assumption is validated.'])
    save('routing.json',out)
    print('UNIONS',[(r['n'],round(r['grouped_read_reduction'],3)) for r in window_rows],flush=True)
    print('CACHES',caches,flush=True)


if __name__ == '__main__':
    run()
