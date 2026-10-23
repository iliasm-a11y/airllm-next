#!/usr/bin/env python3
"""Small held-out WikiText-2 quality probe; no generation/throughput claim.

Prepare downloads only a pinned public test file when --download is explicit.
Evaluate is offline and uses four predeclared windows: 128 scored human tokens
each, preceded by 32 unscored corpus tokens. No chat template/BOS is inserted.
The word-level WikiText preprocessing (<unk>, @-@, etc.) is left unchanged.
This 512-label subset does not establish general quality or pretraining purity.
Corpus bytes and temporary reference logits remain only under ignored work/.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'air_llm'))
MiB = 1024**2
VARIANTS = ('original', 'expert4', 'mixed')
REVISION = 'd5678bc8ac0cdd79dbd5e44d4130271018bcec4e'
SOURCE_PATH = 'word_language_model/data/wikitext-2/test.txt'
SOURCE_URL = f'https://raw.githubusercontent.com/pytorch/examples/{REVISION}/{SOURCE_PATH}'
SOURCE_BYTES = 1256449
SOURCE_SHA256 = 'd790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0'
SOURCE_GIT_BLOB = '6f3fc88e4cd88456ea892971f9f12243e8428624'
CARD_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
CARD_URL = f'https://huggingface.co/datasets/Salesforce/wikitext/raw/{CARD_REVISION}/README.md'
CARD_SHA256 = 'd5aabf145d341366745ac703b5844c442c143943b6c77929481395c822b1c28c'
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json', 'merges.txt', 'vocab.json', 'config.json')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def sources():
    return {str(p.relative_to(ROOT)): digest(p) for p in
        sorted((ROOT/'air_llm/airllm/nextgen').glob('*.py'))+[Path(__file__).resolve()]}


def bounded_download(url, path, limit, expected_sha256):
    request = urllib.request.Request(url, headers={'User-Agent': 'airllm-next-heldout-quality-audit'})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(limit+1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Pinned public download length/digest mismatch')
    Path(path).write_bytes(data)


def corpus_provenance(cache, download=False):
    cache.mkdir(parents=True, exist_ok=True)
    data_path, card_path = cache/'test.txt', cache/'official-dataset-card.md'
    for path, url, limit, expected in [(data_path, SOURCE_URL, 2*MiB, SOURCE_SHA256),
                                      (card_path, CARD_URL, 128*1024, CARD_SHA256)]:
        if not path.is_file():
            if not download:
                raise FileNotFoundError('Pinned corpus/card missing; run --phase prepare --download explicitly')
            bounded_download(url, path, limit, expected)
        if digest(path) != expected:
            raise ValueError(f'Pinned corpus/card digest mismatch: {path.name}')
    raw = data_path.read_bytes()
    git_blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if len(raw) != SOURCE_BYTES or git_blob != SOURCE_GIT_BLOB:
        raise ValueError('Corpus differs from the official pinned Git object')
    provenance_path = cache/'corpus-provenance.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if (provenance.get('source_sha256') != SOURCE_SHA256 or
                provenance.get('source_revision') != REVISION):
            raise ValueError('Corpus provenance differs from the pinned source')
    else:
        provenance = dict(dataset='WikiText-2', split='test',
            variant='PyTorch examples bundled word-level preprocessed WikiText-2; not raw',
            source_url=SOURCE_URL, source_repository='https://github.com/pytorch/examples',
            source_revision=REVISION, source_path=SOURCE_PATH,
            source_bytes=SOURCE_BYTES, source_sha256=SOURCE_SHA256,
            source_git_blob_sha1=SOURCE_GIT_BLOB,
            official_github_tree_api=f'https://api.github.com/repos/pytorch/examples/git/trees/{REVISION}?recursive=1',
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
            license=dict(dataset_licenses=['CC-BY-SA-3.0', 'GFDL'],
                dataset_card_url=CARD_URL, dataset_card_revision=CARD_REVISION,
                dataset_card_sha256=CARD_SHA256,
                attribution='WikiText by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher (Salesforce/MetaMind), derived from Wikipedia contributors; distributed via official PyTorch examples.',
                license_url='https://creativecommons.org/licenses/by-sa/3.0/',
                dataset_page='https://huggingface.co/datasets/Salesforce/wikitext',
                paper='https://arxiv.org/abs/1609.07843'),
            git_content_policy='Corpus bytes remain only under ignored work/; reports contain provenance, hashes, indices and selected token ids.')
        write_json(provenance_path, provenance)
    return raw.decode('utf-8'), provenance


def select_segments(token_ids, count=4, targets=128, context=32):
    """Midpoints of equally spaced corpus quantiles, fixed before evaluation."""
    if min(count, targets, context) < 1 or len(token_ids) < count*(targets+context):
        raise ValueError('Need positive segment/context/target counts and disjoint corpus windows')
    width = targets+context
    rows = []
    for index in range(count):
        start = (len(token_ids)*(2*index+1))//(2*count)-width//2
        values = [int(value) for value in token_ids[start:start+width]]
        if len(values) != width:
            raise AssertionError('Selected corpus span is truncated')
        rows.append(dict(segment=index, source_token_start=start, source_token_end=start+width,
            target_token_start=start+context, target_token_end=start+width,
            context_tokens=context, scored_tokens=targets, token_ids=values,
            token_ids_sha256=hashlib.sha256(np.asarray(values, dtype='<i4').tobytes()).hexdigest()))
    if any(left['source_token_end'] > right['source_token_start'] for left,right in zip(rows, rows[1:])):
        raise ValueError('Selected corpus windows overlap')
    return rows


def prepare(args):
    from transformers import AutoTokenizer
    text, provenance = corpus_provenance(args.corpus_cache, download=args.download)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if any(type(value) is not int or not 0 <= value <= np.iinfo(np.int32).max for value in token_ids):
        raise ValueError('Tokenizer returned invalid source token ids')
    assets = {name: digest(args.model_dir/name) for name in TOKENIZER_FILES if (args.model_dir/name).is_file()}
    manifest = dict(schema_version=1, corpus=provenance, corpus_total_token_count=len(token_ids),
        tokenizer_files_sha256=assets, tokenizer_source=str(args.model_dir),
        tokenization=dict(add_special_tokens=False, chat_template=False, text_normalization='none',
            newline_handling='preserve downloaded UTF-8 bytes', synthetic_bos=False),
        selection='For each equal token-stream quantile, take midpoint-aligned contiguous context+target window; selected before any model inference.',
        score_alignment='For each span, input_ids=token_ids[:-1], labels=token_ids[context_tokens:], retain the last scored_tokens logits.',
        context_reset='Independent source window for every segment; no KV state shared between segments',
        held_out_scope='WikiText-2 test split, not used for conversion, tuning or selection by this experiment; Granite pretraining overlap is unknown.',
        standard_wikitext_perplexity=False,
        segments=select_segments(token_ids, args.segments, args.tokens, args.context_tokens))
    write_json(args.corpus_cache/'segments.json', manifest)
    return manifest


def validate_prepared_manifest(manifest, count, targets, context):
    """Reject accidental edits to prepared identities, lengths or positions."""
    if (manifest.get('corpus', {}).get('source_sha256') != SOURCE_SHA256 or
            manifest.get('corpus', {}).get('source_revision') != REVISION or
            len(manifest.get('segments', [])) != count):
        raise ValueError('Prepared manifest does not identify the pinned test corpus')
    width = targets+context
    previous_end = -1
    for index, row in enumerate(manifest['segments']):
        values = row['token_ids']
        start = (manifest['corpus_total_token_count']*(2*index+1))//(2*count)-width//2
        if (row['segment'] != index or row['scored_tokens'] != targets or row['context_tokens'] != context or
                row['source_token_start'] != start or row['source_token_end'] != start+width or
                row['target_token_start'] != start+context or row['target_token_end'] != start+width or
                start < previous_end or len(values) != width or
                any(type(value) is not int or not 0 <= value <= np.iinfo(np.int32).max for value in values)):
            raise ValueError('Prepared segment counts/positions/token ids are inconsistent')
        if hashlib.sha256(np.asarray(values, dtype='<i4').tobytes()).hexdigest() != row['token_ids_sha256']:
            raise ValueError('Prepared segment token digest mismatch')
        previous_end = start+width


def safe_exp(value):
    return math.exp(value) if value <= math.log(sys.float_info.max) else None


def quality_metrics(reference, actual, labels, row_chunk=8):
    """Bound float64 temporaries to row_chunk*vocabulary, not corpus size."""
    reference, actual, labels = np.asarray(reference), np.asarray(actual), np.asarray(labels)
    if reference.shape != actual.shape or reference.ndim != 2 or not reference.size:
        raise ValueError('Expected equal nonempty [positions, vocabulary] logits')
    if labels.shape != (reference.shape[0],) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Labels must have one integer id for each scored position')
    if np.any(labels < 0) or np.any(labels >= reference.shape[1]) or row_chunk < 1:
        raise ValueError('Invalid labels or row chunk')
    result = dict(scored_tokens=len(labels), vocabulary=reference.shape[1],
        reference_nll_sum=0., variant_nll_sum=0., kl_sum=0., max_kl=0.,
        top1_agreement_count=0, reference_human_top1_correct=0, variant_human_top1_correct=0,
        squared_logit_error_sum=0., max_abs_logit_error=0.)
    for start in range(0, len(labels), row_chunk):
        stop = min(len(labels), start+row_chunk)
        ref = np.asarray(reference[start:stop], dtype=np.float64)
        value = np.asarray(actual[start:stop], dtype=np.float64)
        if not np.isfinite(ref).all() or not np.isfinite(value).all():
            raise ValueError('Nonfinite corpus logits')
        def log_softmax(array):
            centered = array-array.max(axis=-1, keepdims=True)
            return centered-np.log(np.exp(centered).sum(axis=-1, keepdims=True))
        log_ref, log_value = log_softmax(ref), log_softmax(value)
        target = labels[start:stop]
        positions = np.arange(stop-start)
        result['reference_nll_sum'] -= float(log_ref[positions, target].sum())
        result['variant_nll_sum'] -= float(log_value[positions, target].sum())
        kl = np.sum(np.exp(log_ref)*(log_ref-log_value), axis=-1)
        result['kl_sum'] += float(kl.sum())
        result['max_kl'] = max(result['max_kl'], float(kl.max()))
        ref_top, value_top = ref.argmax(axis=-1), value.argmax(axis=-1)
        result['top1_agreement_count'] += int(np.sum(ref_top == value_top))
        result['reference_human_top1_correct'] += int(np.sum(ref_top == target))
        result['variant_human_top1_correct'] += int(np.sum(value_top == target))
        error = value-ref
        result['squared_logit_error_sum'] += float(np.sum(error*error))
        result['max_abs_logit_error'] = max(result['max_abs_logit_error'], float(np.max(np.abs(error))))
    return finalize_metrics(result)


def finalize_metrics(row):
    row = dict(row)
    count = row['scored_tokens']
    reference_nll, variant_nll = row['reference_nll_sum']/count, row['variant_nll_sum']/count
    ref_ppl, value_ppl = safe_exp(reference_nll), safe_exp(variant_nll)
    delta = variant_nll-reference_nll
    row.update(reference_mean_nll=reference_nll, variant_mean_nll=variant_nll,
        mean_nll_delta=delta, reference_perplexity=ref_ppl, variant_perplexity=value_ppl,
        perplexity_delta=value_ppl-ref_ppl if value_ppl is not None and ref_ppl is not None else None,
        perplexity_ratio=safe_exp(delta), mean_kl_reference_to_variant=row['kl_sum']/count,
        top1_agreement_with_original=row['top1_agreement_count']/count,
        reference_human_next_token_accuracy=row['reference_human_top1_correct']/count,
        variant_human_next_token_accuracy=row['variant_human_top1_correct']/count,
        mean_squared_logit_error=row['squared_logit_error_sum']/(count*row['vocabulary']))
    return row


def aggregate(rows):
    if not rows or len({row['vocabulary'] for row in rows}) != 1:
        raise ValueError('Need same-vocabulary quality rows')
    sums = ('scored_tokens', 'reference_nll_sum', 'variant_nll_sum', 'kl_sum',
            'top1_agreement_count', 'reference_human_top1_correct', 'variant_human_top1_correct',
            'squared_logit_error_sum')
    total = {key: sum(row[key] for row in rows) for key in sums}
    total.update(vocabulary=rows[0]['vocabulary'], max_kl=max(row['max_kl'] for row in rows),
                 max_abs_logit_error=max(row['max_abs_logit_error'] for row in rows))
    return finalize_metrics(total)


def configuration(args, metadata, variant):
    from airllm.nextgen.config import NextConfig
    quantized = None if variant == 'original' else str(args.pack_root/f'quantized-granite-{variant}')
    return NextConfig(device=args.device, dtype='float32', cache_bytes=args.cache_mib*MiB,
        io_bytes=32*MiB, kv_bytes=32*MiB, workspace_bytes=256*MiB, reserve_bytes=1536*MiB,
        max_tensor_bytes=32*MiB, row_chunk=2048, prefill_chunk=64, max_context=512,
        policy='adaptive', prefetch=False, spec_prefetch=False, load_tokenizer=False,
        output_tile_cache=True, use_saved_tuning=False, cache_dir=str(metadata), quantized_dir=quantized)


def audit_sidecar(cp, sidecar, variant):
    from airllm.nextgen.sidecar import expert_matrices, is_expert_tensor
    matrices = list(expert_matrices(cp))
    if variant == 'mixed':
        for key, meta in cp.tensors.items():
            if len(meta['shape']) != 2 or not key.endswith('.weight'):
                continue
            if is_expert_tensor(key) or any(word in key for word in
                    ('embed_tokens', 'indexer.', 'indexers_proj', 'mlp.gate.', 'block_sparse_moe.router.', 'kv_b_proj')):
                continue
            matrices.append((key, None, tuple(meta['shape'])))
    expected = {key if expert is None else f'{key}#{expert}': (shape, 4 if is_expert_tensor(key) else 8)
                for key,expert,shape in matrices}
    if set(sidecar.entries) != set(expected):
        raise ValueError('Corpus evaluation requires a complete sidecar for the declared variant')
    for ident, (shape, bits) in expected.items():
        entry = sidecar.entries[ident]
        if tuple(entry['shape']) != shape or entry.get('bits', sidecar.bits) != bits:
            raise ValueError('Wrong corpus sidecar shape or bit precision')
    sidecar.verify(require_digest=True)
    return dict(manifest_sha256=digest(sidecar.path), matrices=len(expected),
        all_entry_sha256_verified=True, group_size=sidecar.group_size,
        integrity_read_bytes=sidecar.stats['integrity_bytes'],
        payload_bytes=sum(entry['nbytes'] for entry in sidecar.entries.values()))


def budget_snapshot(model, output, config, expected_position):
    cache = model.reader.cache
    kv = output.past_key_values
    workspace = model.expert_batch_metrics()['totals']['workspace_peak_bytes']
    row = dict(cache_bytes=cache.bytes, cache_peak_bytes=cache.peak_bytes,
        io_peak_bytes=model.reader.io.peak_bytes, kv_bytes=kv.bytes, kv_peak_bytes=kv.peak_bytes,
        kv_position=kv.position, expected_kv_position=expected_position,
        expert_workspace_reservation_peak_bytes=workspace)
    if cache.bytes > config.cache_bytes or cache.peak_bytes > config.cache_bytes:
        raise AssertionError('Configured corpus cache budget exceeded')
    if row['io_peak_bytes'] > config.io_bytes:
        raise AssertionError('Configured corpus IO budget exceeded')
    if kv.bytes > config.kv_bytes or kv.peak_bytes > config.kv_bytes:
        raise AssertionError('Configured corpus KV budget exceeded')
    if kv.position != expected_position:
        raise AssertionError('Corpus KV position does not equal the actual input length')
    if workspace > config.workspace_bytes:
        raise AssertionError('Configured corpus expert workspace reservation exceeded')
    return row


def evaluate(args, manifest):
    import mlx.core as mx
    from airllm.nextgen.granite_runtime import AirLLMNextGraniteMoeModel
    from airllm.nextgen.storage import OffsetCheckpoint
    from airllm.nextgen.sidecar import QuantizedSidecar
    identity = sources()
    provenance = json.loads((args.model_dir/'download-provenance.json').read_text())
    for entry in provenance['files']:
        if digest(args.model_dir/entry['name']) != entry['sha256']:
            raise ValueError('Original model provenance mismatch')
    for name, expected in manifest['tokenizer_files_sha256'].items():
        if digest(args.model_dir/name) != expected:
            raise ValueError('Tokenizer changed after corpus segments were prepared')
    mx.set_default_device(getattr(mx, args.device))
    report = dict(schema_version=1, experiment='trained-Granite-WikiText2-small-test-quality',
        arguments={key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()},
        primary_quality_campaign=True, generation_benchmark=False,
        corpus=manifest, segments_manifest_sha256=digest(args.corpus_cache/'segments.json'),
        original_checkpoint=provenance, source_sha256=identity,
        host=platform.platform(), libraries={name: importlib.metadata.version(name) for name in
                                           ('mlx', 'numpy', 'transformers', 'safetensors')},
        precision=dict(original='original BF16 values decoded to FP32',
            expert4='all routed experts affine Q4, other weights original; accumulation FP32',
            mixed='all routed experts Q4 and eligible linear matrices Q8; accumulation FP32'),
        lossless_exactness_claim=False, general_corpus_quality_validated=False,
        giant_model_validated=False, physical_external_ssd_measured=False,
        scope='Four predeclared segments, 128 scored human next-token labels each; 512 labels total by default.',
        eos_policy='Teacher forcing on fixed human corpus labels; no generation or artificial post-EOS continuation.',
        timing_scope='Incidental forward wall times include host logit transfer; no throughput or equal-work speed claim.',
        os_cache='Original/sidecar SHA256 audits warm file pages; OS page cache uncontrolled',
        limits=['Subset perplexity is not standard full WikiText-2 perplexity',
            'Contexts reset between windows; only 32 preceding source tokens by default',
            'Word-level preprocessed corpus, not WikiText raw',
            'Held out from this experiment only; possible overlap with Granite pretraining is unknown',
            '512 labels cannot establish general quality, broad domain performance or giant-model behavior'],
        sidecars={}, records=[], summary={})
    def save():
        report['source_unchanged'] = identity == sources()
        write_json(args.output, report)
    with tempfile.TemporaryDirectory(prefix='corpus-quality-', dir=ROOT/'work') as temporary:
        temporary = Path(temporary)
        with OffsetCheckpoint(args.model_dir, temporary/'metadata', max_read_bytes=32*MiB) as cp:
            for variant in args.variants:
                if variant != 'original':
                    sidecar = QuantizedSidecar(args.pack_root/f'quantized-granite-{variant}', cp.fingerprint)
                    report['sidecars'][variant] = audit_sidecar(cp, sidecar, variant)
        order = ['original']+random.Random(20261023).sample(
            [variant for variant in args.variants if variant != 'original'], len(args.variants)-1)
        report['variant_order'] = order
        report['variant_order_scope'] = 'Original first to create reference logits; quality-only comparison, no timing comparison.'
        for variant in order:
            cfg = configuration(args, temporary/'metadata', variant)
            report.setdefault('options', {})[variant] = cfg.as_dict()
            model = AirLLMNextGraniteMoeModel(args.model_dir, next_config=cfg)
            try:
                for segment in manifest['segments']:
                    ids, count = segment['token_ids'], segment['scored_tokens']
                    labels = np.asarray(ids[segment['context_tokens']:], dtype=np.int32)
                    model.reader.drain(); model.reader.cache.clear(); model.reader.compact_cache.clear()
                    begin = time.perf_counter()
                    output = model.forward([ids[:-1]], logits_to_keep=count)
                    values = np.asarray(output.logits)[0].copy()
                    seconds = time.perf_counter()-begin
                    if values.shape != (count, model.config['vocab_size']):
                        raise ValueError('Unexpected retained corpus logits shape')
                    path = temporary/f'segment-{segment["segment"]}-reference.npy'
                    if variant == 'original':
                        np.save(path, values)
                    reference = np.load(path, mmap_mode='r')
                    quality = quality_metrics(reference, values, labels)
                    budgets = budget_snapshot(model, output, cfg, len(ids)-1)
                    row = dict(variant=variant, segment=segment['segment'], quality=quality,
                        incidental_forward_wall_seconds=seconds,
                        reference_logits_npy_sha256=digest(path),
                        actual_logits_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                        workspace_scope='Conservative expert activation/index reservation computed by the runtime; not measured resident allocation.',
                        **budgets)
                    report['records'].append(row)
                    save()
                    print(json.dumps(dict(variant=variant, segment=segment['segment'],
                        scored_tokens=quality['scored_tokens'], mean_nll=quality['variant_mean_nll'],
                        delta=quality['mean_nll_delta'], kl=quality['mean_kl_reference_to_variant'])), flush=True)
                    del output, values, reference
            finally:
                model.close(save_profile=False)
                mx.clear_cache()
            report['summary'][variant] = aggregate([row['quality'] for row in report['records'] if row['variant'] == variant])
            save()
        # Explicit end audit is outside incidental model-forward intervals.
        # A cached prior verification cannot authorize changed files/manifests.
        report['sidecars_final_audit'] = {}
        try:
            with OffsetCheckpoint(args.model_dir, temporary/'metadata', max_read_bytes=32*MiB) as cp:
                for variant, initial in report['sidecars'].items():
                    sidecar = QuantizedSidecar(args.pack_root/f'quantized-granite-{variant}', cp.fingerprint)
                    final = audit_sidecar(cp, sidecar, variant)
                    final['same_identity_as_initial'] = all(final[key] == initial[key] for key in
                        ('manifest_sha256', 'matrices', 'group_size', 'payload_bytes'))
                    report['sidecars_final_audit'][variant] = final
                    if not final['same_identity_as_initial']:
                        raise ValueError('Sidecar identity changed during corpus evaluation')
        except Exception as error:
            report['sidecars_final_audit_error'] = str(error)
            save()
            raise
    report['source_unchanged'] = identity == sources()
    report['original_files_unchanged'] = all(digest(args.model_dir/entry['name']) == entry['sha256'] for entry in provenance['files'])
    report['tokenizer_files_unchanged'] = all(digest(args.model_dir/name) == expected
                                            for name,expected in manifest['tokenizer_files_sha256'].items())
    save()
    if not all(report[key] for key in ('source_unchanged', 'original_files_unchanged', 'tokenizer_files_unchanged')):
        raise SystemExit('Sources/model/tokenizer changed during corpus evaluation; evidence preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'evaluate'], default='evaluate')
    parser.add_argument('--download', action='store_true', help='Only prepare: explicitly permit pinned public corpus/card downloads')
    parser.add_argument('--corpus-cache', type=Path, default=ROOT/'work/wikitext2')
    parser.add_argument('--pack-root', type=Path, default=ROOT/'work')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--cache-mib', type=int, default=256)
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    parser.add_argument('--segments', type=int, default=4)
    parser.add_argument('--tokens', type=int, default=128, help='Scored human labels per segment')
    parser.add_argument('--context-tokens', type=int, default=32, help='Unscored preceding corpus tokens per segment')
    args = parser.parse_args()
    if min(args.segments, args.tokens, args.context_tokens) < 1 or args.cache_mib < 0:
        parser.error('Use positive segments/tokens/context and a nonnegative cache')
    if args.tokens+args.context_tokens > 512:
        parser.error('Corpus span must fit the same 512-position context budget as benchmark F')
    if 'original' not in args.variants or len(set(args.variants)) != len(args.variants):
        parser.error('Include original exactly once and do not repeat variants')
    for name in ('model_dir', 'output', 'corpus_cache', 'pack_root'):
        setattr(args, name, getattr(args, name).resolve())
    if not args.corpus_cache.is_relative_to(ROOT/'work'):
        parser.error('Corpus cache must stay under ignored work/ to keep corpus bytes out of Git')
    if args.output.is_relative_to(args.model_dir) or args.corpus_cache.is_relative_to(args.model_dir):
        parser.error('Reports/corpus must be outside the original model directory')
    if args.download and args.phase != 'prepare':
        parser.error('Evaluate is offline; --download belongs only to --phase prepare')
    if args.phase == 'prepare':
        manifest = prepare(args)
        write_json(args.output, dict(prepared=True, model_inference_run=False,
            corpus_manifest=manifest, segments_manifest_sha256=digest(args.corpus_cache/'segments.json'),
            source_sha256=sources()))
        print(json.dumps(dict(prepared=True, segments=len(manifest['segments']),
            scored_tokens=sum(row['scored_tokens'] for row in manifest['segments']),
            source_sha256=SOURCE_SHA256)), flush=True)
    else:
        corpus_provenance(args.corpus_cache, download=False)
        manifest = json.loads((args.corpus_cache/'segments.json').read_text())
        validate_prepared_manifest(manifest, args.segments, args.tokens, args.context_tokens)
        evaluate(args, manifest)


if __name__ == '__main__':
    main()
