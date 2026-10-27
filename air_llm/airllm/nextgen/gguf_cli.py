"""Explicit GGUF text CLI with owned-process supervision enabled by default."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from time import perf_counter


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer')
    return number


def _string_buffer_size(value):
    number = int(value)
    if not 0 <= number <= 65536:
        raise argparse.ArgumentTypeError('String buffer must be between 0 and 65536 bytes')
    return number


def _run(args):
    from .config import NextConfig
    from .gguf import GGUFReader
    from .gguf_qwen import NativeGGUFQwenModel
    from .gguf_tokenizer import GGUFTokenizer
    begin = perf_counter()
    with GGUFReader(args.model, memory_budget_bytes=256 << 20, max_metadata_bytes=192 << 20,
                    bulk_metadata=args.bulk_metadata, string_buffer_bytes=args.string_buffer_bytes) as checkpoint:
        tokenizer = GGUFTokenizer(checkpoint.metadata)
    del checkpoint
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}])
    if len(ids) + args.max_new_tokens > args.max_context:
        raise MemoryError('Prompt plus requested tokens exceeds max_context before weight loading')
    options = NextConfig(cache_bytes=args.cache_mib << 20, io_bytes=64 << 20, kv_bytes=192 << 20,
        workspace_bytes=128 << 20, reserve_bytes=args.reserve_mib << 20, max_tensor_bytes=16 << 20,
        row_chunk=args.row_chunk, prefill_chunk=args.prefill_chunk, max_context=args.max_context,
        policy=args.cache_policy, device=args.device, dtype='float32', prefetch=False, load_tokenizer=False,
        io_mode=args.io_mode,zero_copy_weights=args.zero_copy_weights)
    with NativeGGUFQwenModel(args.model, options, kernel=args.kernel,cache_decay_interval=args.cache_decay,
            expert_batch=args.expert_batch,concat_tiles=args.concat_tiles,pipeline_tiles=args.pipeline_tiles,
            metal_simd_groups=args.simd_groups,packed_copy_free=args.packed_copy_free,packed_direct_io=args.packed_direct_io,
            bulk_metadata=args.bulk_metadata,string_buffer_bytes=args.string_buffer_bytes,expert_pipeline=args.expert_pipeline) as model:
        output = model.generate([ids], max_new_tokens=args.max_new_tokens, eos_token_id=[248046, 248044])
        generated = output.tolist()[0][len(ids):]
        result = {'text': tokenizer.decode(generated, skip_special_tokens=True), 'input_ids': ids,
                  'generated_ids': generated, 'wall_seconds_including_tokenizer': perf_counter() - begin,
                  'options': asdict(options), 'metrics': model.metrics(),
                  'validation_scope': 'Experimental native text backend; independent validation is described in NATIVE_GGUF_REPORT_FR.md'}
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(result['text'], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='airllm-next', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('gguf-inspect', help='Read GGUF index without mapping weights')
    inspect.add_argument('model', type=Path)
    run = sub.add_parser('gguf-run', help='Experimental supervised native Qwen35 MoE GGUF text generation')
    run.add_argument('model', type=Path)
    run.add_argument('--prompt', default='Bonjour')
    run.add_argument('--max-new-tokens', type=_positive, default=32)
    run.add_argument('--max-context', type=_positive, default=256)
    run.add_argument('--cache-mib', type=_positive, default=256)
    run.add_argument('--cache-decay', type=_positive, help='Experimental adaptive-cache half-life in requests')
    run.add_argument('--cache-policy',choices=['adaptive','lru'],default='adaptive')
    run.add_argument('--expert-batch',action='store_true',help='Batch already-routed packed experts without changing their reduction order')
    run.add_argument('--expert-pipeline',action='store_true',help='Experimental bounded gate/up overlap; requires --expert-batch')
    run.add_argument('--concat-tiles',action='store_true',help='Assemble evaluated row tiles with one concatenation')
    run.add_argument('--pipeline-tiles',action='store_true',help='Experimental two-tile overlap; requires --concat-tiles and --kernel metal')
    run.add_argument('--zero-copy-weights',action='store_true',help='Exact packed-buffer import with a portable copy fallback')
    run.add_argument('--simd-groups',type=int,choices=[1,2,4,8],default=1,help='Independent SIMD row groups per Metal threadgroup; each row reduction is unchanged')
    run.add_argument('--reserve-mib', type=_positive, default=3072)
    run.add_argument('--max-rss-mib', type=_positive, default=2048)
    run.add_argument('--max-swap-growth-mib', type=_positive, default=128)
    run.add_argument('--timeout', type=_positive, default=900)
    run.add_argument('--row-chunk', type=_positive, default=512)
    run.add_argument('--prefill-chunk', type=_positive, default=16)
    run.add_argument('--device', choices=['gpu', 'cpu', 'auto'], default='auto')
    run.add_argument('--kernel', choices=['reference', 'metal'], default='reference')
    transport=run.add_mutually_exclusive_group()
    transport.add_argument('--packed-direct-io', action='store_true', help='Experimental bounded read into one final writable host buffer')
    transport.add_argument('--packed-copy-free', action='store_true', help='Experimental bounded zero-copy packed host reads')
    run.add_argument('--bulk-metadata', action='store_true', help='Experimental bounded scalar-array metadata reads')
    run.add_argument('--string-buffer-bytes', type=_string_buffer_size, default=0,
                     metavar='0..65536', help='Bounded string-array parser buffer; zero uses legacy reads')
    run.add_argument('--io-mode', choices=['pread', 'pread_nocache'], default='pread')
    run.add_argument('--report', type=Path, required=True, help='Local JSON report and adjacent log/guard files')
    run.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw)
    if args.command=='gguf-run':
        if args.pipeline_tiles and (not args.concat_tiles or args.kernel!='metal'):
            parser.error('--pipeline-tiles requires --concat-tiles and --kernel metal')
        if args.expert_batch and args.kernel!='metal':parser.error('--expert-batch requires --kernel metal')
        if args.expert_pipeline and not args.expert_batch:parser.error('--expert-pipeline requires --expert-batch')
        if args.cache_decay is not None and args.cache_policy!='adaptive':parser.error('--cache-decay requires --cache-policy adaptive')
    if args.command == 'gguf-inspect':
        from .gguf import GGUFReader
        with GGUFReader(args.model, include_tokenizer_metadata=False) as reader:
            print(json.dumps({'file_bytes': reader.file_size, 'metadata': reader.metadata,
                              'tensors': {n: {'shape': d.shape, 'quantization': d.dtype, 'bytes': d.nbytes}
                                          for n, d in reader.tensors.items()},
                              'weight_payload_loaded': False}, indent=2))
        return 0
    from .experiment_guard import validate_output_paths
    validate_output_paths(args.report, protected=[args.model])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        _run(args)
        return 0
    from .experiment_guard import ExperimentLimits, supervise
    limits = ExperimentLimits(rss_bytes=args.max_rss_mib << 20, reserve_bytes=args.reserve_mib << 20,
        swap_growth_bytes=args.max_swap_growth_mib << 20, admitted_payload_bytes=(args.cache_mib + 640) << 20,
        timeout_seconds=args.timeout)
    supervise([sys.executable, '-m', 'airllm.nextgen.gguf_cli', *raw, '--worker'], args.report, limits,
              env=dict(os.environ, MLX_ENABLE_TF32='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    print(json.loads(args.report.read_text())['text'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
