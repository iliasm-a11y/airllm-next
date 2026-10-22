"""Build a metric-specific ranking and an evidence manifest; no new benchmark."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from common import ROOT,REPO,digest,save


def load(name):return json.loads((ROOT/'results'/name).read_text())


def main():
    compression=load('compression.json')
    transport=load('transport.json')
    routing=load('routing.json')
    cache_ref=load('cache_reference.json')
    ranks=[]
    # Do not rank incomparable experiments as one global inference speed claim.
    for r in compression['results']:
        ranks.append(dict(metric='encoded_size_ratio_lower_is_better',method=r['codec'],value=r['ratio'],
            evidence='measured Ling BF16 sample',inference_speedup_measured=False))
    for r in transport['summary']:
        ranks.append(dict(metric='warm_transport_speedup_higher_is_better',method=r['mode'],value=r['transport_speedup_vs_offset'],
            evidence='measured OffsetCheckpoint path, warm cache',inference_speedup_measured=False))
    for r in routing['windows']:
        ranks.append(dict(metric='teacher_forced_expert_request_reduction',method='group_'+str(r['n']),value=r['grouped_read_reduction'],
            evidence='inherited OLMoE trace replay',inference_speedup_measured=False))
    with (ROOT/'results/ranking.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ranks[0]),lineterminator='\n');w.writeheader();w.writerows(ranks)
    tests=[]
    for name in ['tests.xml','inherited-controls.xml']:
        suites=ET.parse(ROOT/'results'/name).getroot()
        children=list(suites) if suites.tag=='testsuites' else [suites]
        tests.append(dict(file=name,tests=sum(int(s.get('tests',0)) for s in children),
            errors=sum(int(s.get('errors',0)) for s in children),
            failures=sum(int(s.get('failures',0)) for s in children),
            skipped=sum(int(s.get('skipped',0)) for s in children)))
    summary=dict(base_commit='d43032f38a0325aafd2973847d7ed84b21280c61',
        branch='research/exact-io-2026-10-22',tests=tests,
        passed=sum(t['tests']-t['skipped']-t['failures']-t['errors'] for t in tests),
        complete_historical_mlx_suite_executed=False,
        full_glm_or_qwen_inference_measured=False,external_ssd_measured=False,
        metal_model_runtime_measured=False,
        exact_trained_expert_blocks=4,
        palette_encoded_ratio=next(r['ratio'] for r in compression['results'] if r['codec']=='palette16_c_alloc_hash'),
        native_default_cache_replay=cache_ref['results'],
        rejected_or_unproven=['8.1 seconds/token reproduction','universal 10x impossibility theorem',
            'unconditional speed benefit of lossless transport','XOR inter-expert size benefit',
            'tiny exact low rank on the tested matrices','full-model bitwise parity'])
    numerical_path=ROOT/'results/numerical_contract.json'
    if numerical_path.exists():
        numerical=json.loads(numerical_path.read_text())
        summary['numerical_contract_experiment']={k:numerical[k] for k in [
            'block_comparisons','block_comparisons_with_bit_differences',
            'same_row_order_comparisons_bitwise_equal','full_model_logits_or_tokens_compared']}
    # Retesting the same tests is independent confirmation, not more unique tests.
    resumed=[]
    for name in ['tests-resumed.xml','inherited-controls-resumed.xml']:
        p=ROOT/'results'/name
        if not p.exists():continue
        suites=ET.parse(p).getroot()
        children=list(suites) if suites.tag=='testsuites' else [suites]
        resumed.append(dict(file=name,tests=sum(int(s.get('tests',0)) for s in children),
            errors=sum(int(s.get('errors',0)) for s in children),
            failures=sum(int(s.get('failures',0)) for s in children),
            skipped=sum(int(s.get('skipped',0)) for s in children)))
    summary['resumed_verification']=resumed
    summary['resumed_verification_passed']=sum(t['tests']-t['skipped']-t['failures']-t['errors'] for t in resumed)
    save('validation-summary.json',summary)
    included=[]
    for folder in [ROOT,REPO/'air_llm/airllm/nextgen']:
        for p in sorted(folder.rglob('*')):
            if not p.is_file() or any(x in p.relative_to(folder).parts for x in ['__pycache__','build','samples']):continue
            if p.name=='source-manifest.json':continue
            included.append(dict(file=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=digest(p)))
    extra=REPO/'air_llm/airllm/ultra/checkpoint.py'
    included.append(dict(file=str(extra.relative_to(REPO)),bytes=extra.stat().st_size,sha256=digest(extra)))
    save('source-manifest.json',dict(files=included,note='Manifest excludes itself, build products and locally sampled weight payloads.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
