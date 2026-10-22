"""Explicit hypothetical scenarios. No time below is an end-to-end benchmark."""
import json
from common import ROOT, save


def progress(alpha,K):
    return sum(alpha**i for i in range(K+1))


def expected_union(N,k,n):
    return N*(1-(1-k/N)**n)


def run():
    models=json.loads((ROOT/'results/metadata.json').read_text())
    rows=[]
    for m,N,k,L in [(models[0],256,8,75),(models[1],512,10,92)]:
        A=m['always_stored_bytes_per_pass']
        embedding=m['selective_embedding_row_bytes']
        R=m['active_routed_stored_bytes_per_position']
        cache=10e9
        for rho in [1.0,0.75]:
            for alpha in [0,0.5,0.8,0.95,1]:
                options=[]
                for K in [0,1,2,4,8,16,32,64,128,256,512]:
                    union_factor=expected_union(N,k,K+1)/k
                    traffic=rho*(max(0,A-cache)+embedding*(K+1)+R*union_factor)
                    tau=progress(alpha,K)
                    options.append(dict(K=K,positions=K+1,expected_tokens_emitted=tau,
                        encoded_bytes_per_pass=traffic,ota_bytes=traffic/tau,
                        transfer_only_seconds_per_token_at_3GBps=traffic/tau/3e9,
                        transfer_only_round_latency_seconds_at_3GBps=traffic/3e9))
                best=min(options,key=lambda r:r['ota_bytes'])
                baseline=A+R+embedding
                rows.append(dict(model=m['name'],routing='hypothetical uniform IID',
                    independent_acceptance_alpha=alpha,lossless_ratio_assumed=rho,
                    compact_stored_weight_residency_assumed_bytes=cache,
                    no_cache_naive_transfer_seconds_at_3GBps=baseline/3e9,
                    all_options=options,best=best,
                    best_ota_reduction_vs_no_cache_baseline=baseline/best['ota_bytes']))
    save('scenarios.json',dict(evidence='simulation only',rows=rows,
        omitted_costs=['Draft generation','DSA/attention and KV work','CPU/GPU compute',
        'Decompression and hashing','Allocation and synchronization','First-token/prefill cost'],
        assumptions=['3 GB/s is hypothetical external SSD bandwidth.',
        '10 GB cache stores original compact bytes, unlike the existing decoded-weight cache.',
        'alpha and lossless ratios are assumptions, not measured on GLM or Qwen.',
        'K+1 positions must fit ONE execution chunk for the stated union sharing.'],
        preliminary_claims={'13.8s': 'Reproduced arithmetically as GLM 41.383447104 GB / hypothetical 3 GB/s.',
            '8.1s': 'Not independently reproduced: original parameter set, MTP acceptance and measurements absent.'}))
    print([(r['model'],r['independent_acceptance_alpha'],r['lossless_ratio_assumed'],
        r['best']['K'],round(r['best_ota_reduction_vs_no_cache_baseline'],3)) for r in rows],flush=True)


if __name__=='__main__':
    run()
