"""Inspect locally or run a supervised GLM5-Next text experiment.

Run output is real only when status=generated. Synthetic tests and metadata
inspection never set this status. Token events are flushed into the worker log.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import traceback


def inspect_model(path):
    from .gguf_split import SplitGGUFReader
    from .gguf_glm5 import NativeGGUFGLM5Model, gguf_glm5_config
    with SplitGGUFReader(path, include_tokenizer_metadata=False) as cp:
        config = gguf_glm5_config(cp)
        probe = NativeGGUFGLM5Model.__new__(NativeGGUFGLM5Model)
        probe.config, probe.checkpoint = config, cp
        probe._validate_layout()
        return {'status': 'layout_checked', 'weight_payload_loaded': False,
                'inference_performed': False, 'config': config,
                'shards': [str(p.resolve()) for p in cp.paths],
                'file_bytes': cp.file_size, 'tensor_count': len(cp.tensors),
                'tensor_types': dict(Counter(d.dtype for d in cp.tensors.values())),
                'io': cp.stats}


def _worker(args):
    from .config import NextConfig
    from .physical_io import physical_read_bytes, physical_read_counter_info
    from .ronin_glm53 import diagnostics, load_conversation
    from ..ultra.checkpoint import atomic_json
    result = diagnostics() | {'status': 'starting', 'inference_performed': False}
    model = None
    try:
        # Reuse the same conservative disk policy as the installer.
        from .glm53_download import MIN_RESERVE_BYTES
        # Existing files require no new download; runtime still protects boot
        # volume through memory/swap supervision. No installer runs here.
        boot_free = shutil.disk_usage('/').free
        result['boot_volume_free_bytes'] = boot_free
        if sys.platform == 'darwin' and boot_free < MIN_RESERVE_BYTES:
            raise OSError('Insufficient macOS boot-volume reserve before inference')
        options = NextConfig(cache_bytes=256 << 20, io_bytes=64 << 20,
            kv_bytes=512 << 20, workspace_bytes=256 << 20, reserve_bytes=4 << 30,
            max_tensor_bytes=16 << 20, max_context=args.max_context,
            row_chunk=128, prefill_chunk=1, device=args.device, dtype='float32',
            policy='adaptive', prefetch=False, load_tokenizer=False)
        before = physical_read_bytes()
        model, session = load_conversation(args.model, next_config=options)
        def event(value):
            print(json.dumps({'event': 'token', **value}), flush=True)
        response = session.reply(args.prompt, max_new_tokens=args.max_new_tokens,
                                 on_token=event)
        after = physical_read_bytes()
        result.update(status='generated', inference_performed=True,
                      response=response, metrics=model.metrics(),
                      observed_read_counter=physical_read_counter_info(),
                      observed_read_bytes=(None if before is None or after is None else after-before))
        # A generation by itself does not certify fidelity, M5 or RONIN.
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        raise
    finally:
        if model is not None:
            model.close()
        atomic_json(args.report, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'run'])
    parser.add_argument('model', type=Path, help='Local first GGUF shard; never a Hub ID')
    parser.add_argument('--prompt', default='Bonjour, présente-toi brièvement.')
    parser.add_argument('--max-new-tokens', type=int, default=8)
    parser.add_argument('--max-context', type=int, default=512)
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw)
    if min(args.max_new_tokens, args.max_context) < 1:
        parser.error('Token/context limits must be positive')
    if args.command == 'inspect':
        print(json.dumps(inspect_model(args.model), indent=2))
        return 0
    if args.report is None:
        parser.error('run requires --report for error and guard evidence')
    from .experiment_guard import ExperimentLimits, supervise, validate_output_paths
    # Index only, before any output can overwrite a checkpoint shard.
    from .gguf_split import SplitGGUFReader
    with SplitGGUFReader(args.model, include_tokenizer_metadata=False) as cp:
        validate_output_paths(args.report, protected=cp.paths)
    if any(p.exists() for p in (args.report, args.report.with_suffix('.log'),
                                args.report.with_suffix('.guard.json'))) and not args.worker:
        parser.error('Use a new report filename; existing evidence is preserved')
    if args.worker:
        _worker(args)
        return 0
    limits = ExperimentLimits(rss_bytes=3 << 30, reserve_bytes=4 << 30,
        swap_growth_bytes=128 << 20, admitted_payload_bytes=1536 << 20,
        timeout_seconds=args.timeout)
    supervise([sys.executable, '-m', 'airllm.nextgen.glm53_cli', *raw, '--worker'],
        args.report, limits, env=dict(os.environ, MLX_ENABLE_TF32='0',
                                    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    result = json.loads(args.report.read_text())
    print(result['response']['text'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
