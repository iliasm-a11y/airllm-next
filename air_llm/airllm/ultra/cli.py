"""Local-only launcher, compatibility audit and disposable SSD benchmark."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time


def doctor(args):
    import psutil
    import torch
    import transformers
    path = Path(args.directory).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    report = {'platform': platform.platform(), 'machine': platform.machine(),
              'python': sys.version, 'torch': torch.__version__,
              'transformers': transformers.__version__,
              'mps_available': torch.backends.mps.is_available(),
              'cuda_available': torch.cuda.is_available(),
              'memory_total_bytes': psutil.virtual_memory().total,
              'memory_available_bytes': psutil.virtual_memory().available,
              'directory': str(path), 'disk_free_bytes': shutil.disk_usage(path).free,
              'tested_hardware': 'See benchmarks/ and docs/VALIDATION.md; detection is not validation'}
    print(json.dumps(report, indent=2))


def audit(args):
    import transformers
    root = Path(args.model).expanduser().resolve()
    cfg = json.loads((root / 'config.json').read_text())
    index_path = root / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    files = sorted(set(index['weight_map'].values())) if index else [p.name for p in root.glob('*.safetensors')]
    report = {'model_type': cfg.get('model_type'), 'architectures': cfg.get('architectures'),
              'layers': cfg.get('num_hidden_layers'),
              'routed_experts': cfg.get('n_routed_experts', cfg.get('num_local_experts')),
              'active_experts_per_token': cfg.get('num_experts_per_tok'),
              'quantization': cfg.get('quantization_config', {}).get('quant_method'),
              'checkpoint_bytes_from_index': index.get('metadata', {}).get('total_size') if index else None,
              'checkpoint_files': len(files), 'missing_files': [f for f in files if not (root / f).is_file()],
              'installed_transformers': transformers.__version__,
              'checkpoint_transformers_version': cfg.get('transformers_version'),
              'full_model_hardware_validation': False,
              'note': 'Config/index inspection does not establish successful inference or device compatibility.'}
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        report['native_transformers_config_recognized'] = True
    except Exception as e:
        report['native_transformers_config_recognized'] = False
        report['error'] = str(e)
    print(json.dumps(report, indent=2))


def io_bench(args):
    import numpy as np
    directory = Path(args.directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    if not 1 <= args.size_mib <= 8192 or not 1 <= args.repeats <= 20:
        raise ValueError('Use 1..8192 MiB and 1..20 repeats')
    size = args.size_mib * 1024**2
    if shutil.disk_usage(directory).free < size + 128*1024**2:
        raise OSError('Insufficient free space for disposable I/O benchmark')
    rng = np.random.default_rng(521)
    block = rng.integers(0, 256, 1024**2, dtype=np.uint8).tobytes()
    fd, filename = tempfile.mkstemp(prefix='airllm-io-', suffix='.tmp', dir=directory)
    try:
        begin = time.perf_counter()
        with os.fdopen(fd, 'wb') as f:
            for _ in range(args.size_mib):
                f.write(block)
            f.flush()
            os.fsync(f.fileno())
        write_seconds = time.perf_counter() - begin
        trials = []
        for _ in range(args.repeats):
            with open(filename, 'rb', buffering=0) as f:
                # Linux can advise discard; macOS uses F_NOCACHE on this temporary
                # file only. Neither guarantees a fully cold device/controller cache.
                method = 'buffered OS cache; not cold SSD'
                if hasattr(os, 'posix_fadvise'):
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    method = 'POSIX_FADV_DONTNEED advice (not guaranteed cold)'
                elif sys.platform == 'darwin':
                    import fcntl
                    fcntl.fcntl(f.fileno(), 48, 1)  # F_NOCACHE, Darwin fcntl.h
                    method = 'Darwin F_NOCACHE (not guaranteed cold)'
                read = 0
                start = time.perf_counter()
                while data := f.read(4*1024**2):
                    read += len(data)
                elapsed = time.perf_counter() - start
                trials.append({'bytes': read, 'seconds': elapsed,
                               'MiB_per_second': read / 1024**2 / elapsed, 'method': method})
        print(json.dumps({'size_bytes': size, 'write_seconds_fsync': write_seconds,
                          'write_MiB_per_second': args.size_mib / write_seconds,
                          'read_trials': trials, 'directory': str(directory)}, indent=2))
    finally:
        os.unlink(filename)


def run(args):
    import torch
    from .config import UltraConfig
    cfg = UltraConfig(cache_bytes=args.cache_mib*1024**2,
                      device_cache_bytes=args.device_cache_mib*1024**2,
                      prefetch_bytes=args.prefetch_mib*1024**2,
                      max_group_bytes=args.max_group_mib*1024**2,
                      reserve_bytes=args.reserve_mib*1024**2,
                      cache_dir=args.cache_dir, prefetch=not args.no_prefetch)
    start = time.perf_counter()
    if args.backend == 'mlx-ultra':
        from .mlx_runtime import MlxLlamaUltra
        with MlxLlamaUltra(args.model, ultra=cfg) as model:
            import mlx.core as mx
            construction = time.perf_counter() - start
            ids = model.tokenizer.encode(args.prompt, add_special_tokens=True)
            input_ids = mx.array([ids], dtype=mx.int32)
            tokens = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
            new_ids = tokens.tolist()[0][len(ids):]
            print(model.tokenizer.decode(new_ids, skip_special_tokens=True))
            metrics = {'backend': args.backend, 'construction_seconds': construction,
                       'total_seconds': time.perf_counter() - start, 'new_tokens': len(new_ids),
                       'hardware_validation': 'This run only; compare reference with tests_ultra/test_mlx.py'}
    else:
        from .runtime import AirLLMUltraModel
        dtype = getattr(torch, args.dtype) if args.dtype else None
        device = None if args.device == 'auto' else args.device
        with AirLLMUltraModel(args.model, device=device, dtype=dtype, ultra=cfg) as model:
            construction = time.perf_counter() - start
            ids = model.tokenizer(args.prompt, return_tensors='pt')['input_ids'].to(model.device)
            if ids.shape[-1] + args.max_new_tokens > args.max_context:
                raise ValueError('Prompt + new tokens exceeds --max-context (KV/activation guard)')
            current, kv, output = ids, None, []
            eos = getattr(model.generation_config, 'eos_token_id', None) or model.config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            latencies = []
            ttft = None
            with torch.inference_mode():
                for _ in range(args.max_new_tokens):
                    begin = time.perf_counter()
                    result = model(current, past_key_values=kv, use_cache=True, logits_to_keep=1)
                    token = result.logits[:, -1].argmax(-1)[:, None]
                    value = token.item()  # also synchronizes device work for the measurement
                    kv, current = result.past_key_values, token
                    output.append(value)
                    latencies.append(time.perf_counter() - begin)
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    if value in eos:
                        break
            print(model.tokenizer.decode(output, skip_special_tokens=True))
            metrics = {**model.metrics(), 'backend': 'ultra', 'device': str(model.device),
                       'construction_seconds': construction, 'ttft_including_construction_seconds': ttft,
                       'token_seconds': latencies, 'new_tokens': len(output),
                       'decode_tokens_per_second': ((len(latencies)-1)/sum(latencies[1:])) if len(latencies)>1 else None}
    if args.metrics:
        Path(args.metrics).write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('doctor')
    d.add_argument('--directory', default='.')
    d.set_defaults(func=doctor)
    a = sub.add_parser('audit')
    a.add_argument('--model', required=True)
    a.set_defaults(func=audit)
    io = sub.add_parser('io-bench')
    io.add_argument('--directory', required=True)
    io.add_argument('--size-mib', type=int, default=256)
    io.add_argument('--repeats', type=int, default=3)
    io.set_defaults(func=io_bench)
    r = sub.add_parser('run')
    r.add_argument('--model', required=True)
    r.add_argument('--prompt', default='Bonjour, présente-toi brièvement.')
    r.add_argument('--backend', choices=['ultra', 'mlx-ultra'], default='ultra')
    r.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda:0'])
    r.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'])
    r.add_argument('--max-new-tokens', type=int, default=32)
    r.add_argument('--max-context', type=int, default=512)
    r.add_argument('--cache-mib', type=int, default=512)
    r.add_argument('--device-cache-mib', type=int, default=256)
    r.add_argument('--prefetch-mib', type=int, default=256)
    r.add_argument('--max-group-mib', type=int, default=2048)
    r.add_argument('--reserve-mib', type=int, default=2048)
    r.add_argument('--no-prefetch', action='store_true')
    r.add_argument('--cache-dir')
    r.add_argument('--metrics')
    r.set_defaults(func=run)
    args = p.parse_args(argv)
    if getattr(args, 'max_new_tokens', 1) < 1:
        p.error('--max-new-tokens must be positive')
    args.func(args)


if __name__ == '__main__':
    main()
