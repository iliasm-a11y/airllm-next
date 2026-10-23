#!/usr/bin/env python3
"""Prepare or run a finer quantization probe on the unchanged prior 512 labels.

Default phase `plan` reads metadata and writes a plan; no conversion/inference.
Convert and evaluate are separate explicit phases. Original files, old sidecars,
and the prepared corpus are inputs only. Incidental timings are not throughput.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'air_llm'))
import benchmark_quantized_corpus_quality as corpus
from benchmark_quantized_granite import expected_entries as legacy_entries

MiB=1024**2
RECIPES={
    'expert8':dict(scope='experts',expert_bits=8,linear_bits=8,group_size=64),
    'mixed8':dict(scope='mixed',expert_bits=8,linear_bits=8,group_size=64),
    'expert4g32':dict(scope='experts',expert_bits=4,linear_bits=8,group_size=32),
    'mixed4g32':dict(scope='mixed',expert_bits=4,linear_bits=8,group_size=32),
}


def source_hashes():
    paths=list((ROOT/'air_llm/airllm/nextgen').glob('*.py'))
    paths += [Path(__file__),Path(corpus.__file__),HERE/'benchmark_quantized_granite.py']
    return {str(p.relative_to(ROOT)):corpus.digest(p) for p in sorted(paths)}


def expected_entries(cp,variant):
    from airllm.nextgen.sidecar import is_expert_tensor
    recipe=RECIPES[variant]
    return {ident:(key,expert,shape,recipe['expert_bits'] if is_expert_tensor(key) else recipe['linear_bits'])
        for ident,(key,expert,shape,_) in legacy_entries(cp,recipe['scope']).items()}


def payload_bytes(entries,group_size):
    total=0
    for _,_,shape,bits in entries.values():
        rows,columns=shape
        if columns%group_size or columns%(32//bits):
            raise ValueError('Quantization recipe does not divide a source matrix width')
        total += rows*(columns//(32//bits)+2*columns//group_size)*4
    return total


def validate_sidecar(cp,sidecar,variant):
    expected=expected_entries(cp,variant);recipe=RECIPES[variant]
    if sidecar.group_size!=recipe['group_size'] or sidecar.bits!=recipe['expert_bits']:
        raise ValueError('Wrong quantization recipe in sidecar')
    if set(sidecar.entries)!=set(expected):raise ValueError('Incomplete or wrong-scope sidecar')
    for ident,(_,_,shape,bits) in expected.items():
        entry=sidecar.entries[ident]
        if tuple(entry['shape'])!=shape or entry.get('bits',sidecar.bits)!=bits:
            raise ValueError('Wrong sidecar shape or per-matrix precision')
    sidecar.verify(require_digest=True)
    size=sum(entry['nbytes'] for entry in sidecar.entries.values())
    if size!=payload_bytes(expected,recipe['group_size']):raise ValueError('Unexpected sidecar payload size')
    return dict(path=str(sidecar.root.resolve()),manifest_sha256=corpus.digest(sidecar.path),
        matrices=len(expected),payload_bytes=size,recipe=recipe,
        all_entry_sha256_verified=True,integrity_read_bytes=sidecar.stats['integrity_bytes'])


def paired_prior(prior,manifest,summary,scored_tokens):
    # File byte identity is checked at input time; this guards direct callers.
    if prior.get('corpus')!=manifest:raise ValueError('Prior result uses different corpus windows')
    return dict(same_512_label_protocol=scored_tokens==512,
        scope='Prior report is historical context; the current original is recomputed in this run.',
        previous_summary=prior['summary'],current_summary=summary,
        current_reference_matches_prior_nll=(None if scored_tokens!=512 or 'original' not in summary else
            abs(summary['original']['variant_mean_nll']-prior['summary']['original']['variant_mean_nll'])<1e-5))


def read_inputs(args):
    prior=json.loads(args.prior_report.read_text())
    if args.model_dir is None:args.model_dir=Path(prior['arguments']['model_dir']).resolve()
    manifest_path=args.corpus_manifest or Path(prior['arguments']['corpus_cache'])/'segments.json'
    args.corpus_manifest=manifest_path.resolve()
    if corpus.digest(args.corpus_manifest)!=prior['segments_manifest_sha256']:
        raise ValueError('Prepared corpus manifest no longer matches the prior report')
    manifest=json.loads(args.corpus_manifest.read_text())
    corpus.validate_prepared_manifest(manifest,4,128,32)
    if prior.get('corpus')!=manifest:raise ValueError('Prior report and current corpus manifest disagree')
    corpus_dir=args.corpus_manifest.parent
    for name,expected in [('test.txt',corpus.SOURCE_SHA256),('official-dataset-card.md',corpus.CARD_SHA256)]:
        if corpus.digest(corpus_dir/name)!=expected:raise ValueError('Pinned corpus/card changed')
    assets={name:corpus.digest(args.model_dir/name) for name in manifest['tokenizer_files_sha256']}
    if assets!=manifest['tokenizer_files_sha256']:raise ValueError('Tokenizer differs from prior labels')
    provenance=json.loads((args.model_dir/'download-provenance.json').read_text())
    if provenance!=prior['original_checkpoint']:raise ValueError('Model provenance differs from prior quality probe')
    inputs={str(p):corpus.digest(p) for p in (args.prior_report,args.corpus_manifest,
        corpus_dir/'test.txt',corpus_dir/'official-dataset-card.md',args.model_dir/'download-provenance.json')}
    inputs.update({str(args.model_dir/name):value for name,value in assets.items()})
    return prior,manifest,provenance,inputs


def configuration(args,metadata,variant):
    original=corpus.configuration(args,metadata,'original')
    return replace(original,quantized_dir=None if variant=='original' else
        str(args.pack_root/f'quantized-granite-{variant}'))


def run(args):
    from airllm.nextgen.storage import OffsetCheckpoint
    from airllm.nextgen.sidecar import QuantizedSidecar,convert_experts
    prior,manifest,provenance,inputs=read_inputs(args)
    identity=source_hashes()
    report=dict(schema_version=1,experiment='velocity-finer-quantization-quality',
        phase=args.phase,conversion_run=False,model_inference_run=False,generation_benchmark=False,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        corpus_manifest_sha256=corpus.digest(args.corpus_manifest),
        corpus=manifest,selected_segment_ids=[s['segment'] for s in manifest['segments'][:args.segments]],
        scored_tokens=args.segments*128,original_checkpoint=provenance,
        input_files_sha256=inputs,source_sha256=identity,planned_variants={},sidecars={},records=[],summary={},
        limits=['Subset perplexity is not full WikiText-2 perplexity',
            '512 fixed labels or an explicitly smaller subset cannot establish broad quality',
            'Teacher-forced human-label quality is not a generation-speed measurement',
            'Original embeddings and routers remain unquantized; mixed8 means eligible linear weights only',
            'Original Granite weights fit host RAM; no giant-model or external-SSD validation'])
    def save():
        report['source_unchanged']=identity==source_hashes()
        report['input_files_unchanged']=all(corpus.digest(Path(p))==h for p,h in inputs.items())
        corpus.write_json(args.output,report)
    work=ROOT/'work';work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='velocity-quality-',dir=work) as tmp:
        tmp=Path(tmp)
        with OffsetCheckpoint(args.model_dir,tmp/'metadata',max_read_bytes=32*MiB) as cp:
            for variant in args.variants:
                entries=expected_entries(cp,variant);recipe=RECIPES[variant]
                report['planned_variants'][variant]=dict(recipe=recipe,matrices=len(entries),
                    payload_bytes=payload_bytes(entries,recipe['group_size']),
                    destination=str(args.pack_root/f'quantized-granite-{variant}'))
            report['disk_free_bytes']=shutil.disk_usage(work).free
            report['checkpoint_fingerprint']=cp.fingerprint
            if args.phase=='plan':save();return report
            # A complete source content audit is intentionally deferred until a
            # real run; planning never scans gigabytes during another campaign.
            if not all(corpus.digest(args.model_dir/e['name'])==e['sha256'] for e in provenance['files']):
                raise ValueError('Original checkpoint content digest mismatch')
            for variant in args.variants:
                recipe=RECIPES[variant];destination=args.pack_root/f'quantized-granite-{variant}'
                if args.phase=='convert':
                    required=report['planned_variants'][variant]['payload_bytes']+2*1024**3
                    if shutil.disk_usage(work).free<required:raise MemoryError('Insufficient disk space plus 2 GiB reserve')
                    sidecar=convert_experts(cp,json.loads((args.model_dir/'config.json').read_text()),destination,
                        bits=recipe['expert_bits'],group_size=recipe['group_size'],row_chunk=128,
                        scope=recipe['scope'],linear_bits=recipe['linear_bits'])
                    report['conversion_run']=True
                else:sidecar=QuantizedSidecar(destination,cp.fingerprint)
                report['sidecars'][variant]=validate_sidecar(cp,sidecar,variant);save()
        if args.phase=='convert':return report
        import mlx.core as mx
        from airllm.nextgen.granite_runtime import AirLLMNextGraniteMoeModel
        from airllm.nextgen.tuning import telemetry
        mx.set_default_device(getattr(mx,args.device))
        report['host']=platform.platform()
        report['libraries']={name:importlib.metadata.version(name) for name in ('mlx','numpy','transformers')}
        report['model_inference_run']=True;report['telemetry_before']=telemetry()
        for variant in ['original']+args.variants:
            cfg=configuration(args,tmp/f'metadata-{variant}',variant)
            report.setdefault('options',{})[variant]=cfg.as_dict()
            with AirLLMNextGraniteMoeModel(args.model_dir,next_config=cfg) as model:
                for segment in manifest['segments'][:args.segments]:
                    ids=segment['token_ids'];count=segment['scored_tokens']
                    model.reader.drain();model.reader.cache.clear();model.reader.compact_cache.clear()
                    begin=time.perf_counter()
                    output=model.forward([ids[:-1]],logits_to_keep=count)
                    values=np.asarray(output.logits)[0].copy();elapsed=time.perf_counter()-begin
                    path=tmp/f'segment-{segment["segment"]}-reference.npy'
                    if variant=='original':np.save(path,values)
                    reference=np.load(path,mmap_mode='r')
                    labels=np.asarray(ids[segment['context_tokens']:],dtype=np.int32)
                    quality=corpus.quality_metrics(reference,values,labels)
                    report['records'].append(dict(variant=variant,segment=segment['segment'],quality=quality,
                        incidental_forward_wall_seconds=elapsed,reference_logits_sha256=corpus.digest(path),
                        actual_logits_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                        telemetry=telemetry(),**corpus.budget_snapshot(model,output,cfg,len(ids)-1)))
                    save();print(json.dumps(dict(variant=variant,segment=segment['segment'],
                        scored_tokens=count,nll_delta=quality['mean_nll_delta'],ppl_ratio=quality['perplexity_ratio'])),flush=True)
                    del output,values,reference
                model.close(save_profile=False)
            mx.clear_cache()
            report['summary'][variant]=corpus.aggregate([r['quality'] for r in report['records'] if r['variant']==variant])
            save()
        report['telemetry_after']=telemetry()
        report['prior_comparison']=paired_prior(prior,manifest,report['summary'],report['scored_tokens'])
        with OffsetCheckpoint(args.model_dir,tmp/'metadata-final',max_read_bytes=32*MiB) as cp:
            report['sidecars_final_audit']={}
            for variant,initial in report['sidecars'].items():
                final=validate_sidecar(cp,QuantizedSidecar(initial['path'],cp.fingerprint),variant)
                if final['manifest_sha256']!=initial['manifest_sha256']:raise ValueError('Sidecar changed during inference')
                report['sidecars_final_audit'][variant]=final
        report['original_files_unchanged']=all(corpus.digest(args.model_dir/e['name'])==e['sha256'] for e in provenance['files'])
        save()
        if not all(report[key] for key in ('source_unchanged','input_files_unchanged','original_files_unchanged')):
            raise RuntimeError('Quality experiment identity changed; evidence retained')
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--corpus-manifest',type=Path)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--phase',choices=('plan','convert','evaluate'),default='plan')
    parser.add_argument('--pack-root',type=Path,default=ROOT/'work/velocity-quantization')
    parser.add_argument('--variants',nargs='+',choices=tuple(RECIPES),default=['expert8'])
    parser.add_argument('--segments',type=int,choices=(1,4),default=4)
    parser.add_argument('--device',choices=('cpu','gpu'),default='cpu')
    parser.add_argument('--cache-mib',type=int,default=256)
    args=parser.parse_args()
    for name in ('prior_report','output','corpus_manifest','model_dir','pack_root'):
        if getattr(args,name) is not None:setattr(args,name,getattr(args,name).resolve())
    if len(set(args.variants))!=len(args.variants) or args.cache_mib<0:parser.error('Use unique variants and a nonnegative cache')
    if not args.pack_root.is_relative_to(ROOT/'work'):parser.error('New sidecars must stay under ignored work/')
    prior=json.loads(args.prior_report.read_text())
    model=args.model_dir or Path(prior['arguments']['model_dir']).resolve()
    if args.output.is_relative_to(model) or args.pack_root.is_relative_to(model):parser.error('Keep outputs outside the original model')
    if args.output.exists():parser.error('Output exists; choose a fresh evidence filename')
    run(args)


if __name__=='__main__':main()
