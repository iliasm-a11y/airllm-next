#!/usr/bin/env python3
"""Bounded real GLM-5.3 FP8 subrange validation; never download a full shard.

Acquisition requires HTTP 206 and exact Content-Range before reading a body.
The <=16 MiB response-body allowance includes headers, FP8 tiles and scales.
Samples remain outside Git. Only provenance/hashes/results belong in evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import struct
import sys
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[2]
REVISION = 'ba32f8c19c8738e2c57c71798c601d477a7bf844'
MODEL = 'zai-org/GLM-5.3'
MAX_BODY_BYTES = 16 * 1024**2
MAX_TILE_BYTES = 4 * 1024**2
TILES = (
    ('model.layers.3.mlp.experts.0.gate_proj.weight', 128, 256),
    ('model.layers.3.mlp.experts.0.down_proj.weight', 256, 384),
)
# Fixed before acquisition; FP8 decode is required to be bitwise exact in F32.
LINEAR_RTOL = 2e-5
LINEAR_ATOL = 2e-5


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Ranges:
    def __init__(self):
        self.used = 0
        self.requests = []
        self.resource_sizes = {}

    def read(self, url, start, length):
        if type(start) is not int or type(length) is not int or start < 0 or length < 1:
            raise ValueError('Invalid byte range')
        # Reserve the one-byte overlong-response probe too, even for a server
        # that omits Content-Length. No body read can exceed the total cap.
        if self.used + length + 1 > MAX_BODY_BYTES:
            raise MemoryError('Acquisition response-body allowance exhausted')
        end = start + length - 1
        request = Request(url, headers={'Range': f'bytes={start}-{end}',
                                       'Accept-Encoding': 'identity'})
        with urlopen(request, timeout=30) as response:
            # Never read a 200 body; range-ignoring servers must not stream GBs.
            if response.status != 206:
                raise RuntimeError(f'Rejected HTTP {response.status} before body read')
            value = response.headers.get('Content-Range', '')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', value)
            if not match or tuple(map(int, match.groups()[:2])) != (start, end):
                raise RuntimeError(f'Rejected nonexact Content-Range: {value!r}')
            total = int(match.group(3))
            if total <= end:
                raise RuntimeError('Content-Range exceeds resource size')
            if url in self.resource_sizes and self.resource_sizes[url] != total:
                raise RuntimeError('Resource size changed between pinned range requests')
            self.resource_sizes[url] = total
            declared = response.headers.get('Content-Length')
            if declared is not None and int(declared) != length:
                raise RuntimeError('Rejected nonexact Content-Length')
            data = response.read(length + 1)
            self.used += len(data)
            if len(data) != length:
                raise EOFError('Range response did not contain exactly the requested bytes')
        self.requests.append({'url': url, 'range_start': start, 'range_end': end,
                              'http_status': 206, 'content_range': value,
                              'body_bytes': length, 'resource_bytes': total,
                              'sha256': digest(data)})
        return data


def acquire(directory):
    directory.mkdir(parents=True, exist_ok=True)
    config_path = REPO / 'evidence-v5/sources/glm53-config.json'
    index_path = REPO / 'evidence/glm-5.3.index.json'
    summary = json.loads((REPO / 'evidence-v5/sources/glm53-index-summary.json').read_text())
    if summary['revision'] != REVISION or digest(index_path.read_bytes()) != summary['index_sha256']:
        raise ValueError('Cached official index is not the pinned verified index')
    index = json.loads(index_path.read_text())
    config = json.loads(config_path.read_text())
    if config['quantization_config']['weight_block_size'] != [128, 128]:
        raise ValueError('Sample protocol requires official 128x128 FP8 blocks')
    source = Ranges()
    headers, tiles = {}, []
    for number, (key, first, last) in enumerate(TILES):
        scale_key = key + '_scale_inv'
        shard = index['weight_map'][key]
        if index['weight_map'][scale_key] != shard:
            raise ValueError('Protocol expects matrix and scales in one shard')
        url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{shard}'
        if shard not in headers:
            first8 = source.read(url, 0, 8)
            length = struct.unpack('<Q', first8)[0]
            if length < 2 or length > 1024**2:
                raise ValueError('Safetensors header exceeds 1 MiB allowance')
            raw_header = source.read(url, 8, length)
            header = json.loads(raw_header)
            (directory / (shard + '.header.json')).write_bytes(raw_header)
            headers[shard] = {'length': length, 'sha256': digest(raw_header), 'header': header}
        info = headers[shard]
        tensor, scale = info['header'][key], info['header'][scale_key]
        shape = tensor['shape']
        if tensor['dtype'] not in ('F8_E4M3', 'F8_E4M3FN') or scale['dtype'] != 'F32':
            raise ValueError('Unexpected trained matrix/scales encoding')
        if len(shape) != 2 or not 0 <= first < last <= shape[0] or first % 128 or last % 128:
            raise ValueError('Sample matrix rows must align to official scale blocks')
        if scale['shape'] != [(shape[0] + 127) // 128, (shape[1] + 127) // 128]:
            raise ValueError('Unexpected FP8 block scale dimensions')
        if tensor['data_offsets'][1] - tensor['data_offsets'][0] != shape[0] * shape[1]:
            raise ValueError('Matrix header shape/byte count mismatch')
        if scale['data_offsets'][1] - scale['data_offsets'][0] != 4 * scale['shape'][0] * scale['shape'][1]:
            raise ValueError('Scale header shape/byte count mismatch')
        tile_bytes = (last - first) * shape[1]
        if tile_bytes > MAX_TILE_BYTES:
            raise MemoryError('FP8 sample tile exceeds 4 MiB')
        base = 8 + info['length']
        offset = base + tensor['data_offsets'][0] + first * shape[1]
        raw = source.read(url, offset, tile_bytes)
        scale_first, scale_last = first // 128, last // 128
        scale_offset = base + scale['data_offsets'][0] + 4 * scale_first * scale['shape'][1]
        scale_bytes = 4 * (scale_last - scale_first) * scale['shape'][1]
        raw_scale = source.read(url, scale_offset, scale_bytes)
        matrix_file, scale_file = f'tile-{number}.fp8', f'tile-{number}.scales-f32'
        (directory / matrix_file).write_bytes(raw)
        (directory / scale_file).write_bytes(raw_scale)
        tiles.append({'key': key, 'source_shard': shard, 'source_header_sha256': info['sha256'],
                      'original_shape': shape, 'original_rows': [first, last],
                      'matrix_shape': [last - first, shape[1]],
                      'scale_shape': [scale_last - scale_first, scale['shape'][1]],
                      'scale_row_offset': scale_first, 'block_shape': [128, 128],
                      'matrix_source_offset': offset, 'scale_source_offset': scale_offset,
                      'matrix_file': matrix_file, 'scale_file': scale_file,
                      'matrix_bytes': len(raw), 'scale_bytes': len(raw_scale),
                      'matrix_sha256': digest(raw), 'scale_sha256': digest(raw_scale)})
    manifest = {'model': MODEL, 'revision': REVISION, 'index_sha256': summary['index_sha256'],
                'config_sha256': digest(config_path.read_bytes()), 'tiles': tiles,
                'range_requests': source.requests, 'total_response_body_bytes': source.used,
                'max_response_body_bytes': MAX_BODY_BYTES, 'full_shard_downloaded': False,
                'checkpoint_generation_performed': False}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def reconstruct(directory, manifest):
    """Rebase complete block-aligned source rows, retaining exact source bytes."""
    root = directory / 'minimal-checkpoint'
    root.mkdir(exist_ok=True)
    config_source = REPO / 'evidence-v5/sources/glm53-config.json'
    if digest(config_source.read_bytes()) != manifest['config_sha256']:
        raise ValueError('Pinned config hash changed')
    (root / 'config.json').write_bytes(config_source.read_bytes())
    header, parts, offset = {}, [], 0
    for tile in manifest['tiles']:
        for key, filename, expected, shape, dtype in (
                (tile['key'], tile['matrix_file'], tile['matrix_sha256'], tile['matrix_shape'], 'F8_E4M3'),
                (tile['key'] + '_scale_inv', tile['scale_file'], tile['scale_sha256'], tile['scale_shape'], 'F32')):
            data = (directory / filename).read_bytes()
            if digest(data) != expected:
                raise ValueError('Downloaded sample hash changed')
            header[key] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + len(data)]}
            parts.append(data)
            offset += len(data)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    (root / 'samples.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + b''.join(parts))
    return root


def validate(directory):
    # Environment must be set before MLX initializes any kernels.
    os.environ.setdefault('MLX_ENABLE_TF32', '0')
    sys.path.insert(0, str(REPO / 'air_llm'))
    import numpy as np
    import torch
    import mlx.core as mx
    from dataclasses import replace
    from airllm.nextgen.config import NextConfig
    from airllm.nextgen.storage import OffsetCheckpoint
    from airllm.nextgen.weights import NativeWeights
    torch.set_num_threads(1)
    mx.set_default_device(mx.cpu)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['revision'] != REVISION or manifest['total_response_body_bytes'] > MAX_BODY_BYTES:
        raise ValueError('Sample manifest violates acquisition protocol')
    root = reconstruct(directory, manifest)
    config = json.loads((root / 'config.json').read_text())
    options = replace(NextConfig(), dtype='float32', device='cpu', prefetch=False,
                      cache_bytes=8 * 1024**2, io_bytes=8 * 1024**2,
                      max_tensor_bytes=8 * 1024**2, cache_dir=str(directory / 'metadata'))
    checkpoint = OffsetCheckpoint(root, cache_dir=options.cache_dir,
                                  max_read_bytes=options.max_tensor_bytes)
    reader = NativeWeights(checkpoint, mx, options, config)
    rows = []
    try:
        for tile in manifest['tiles']:
            key = tile['key']
            shape = tile['matrix_shape']
            raw = (directory / tile['matrix_file']).read_bytes()
            scales = (directory / tile['scale_file']).read_bytes()
            reconstructed_raw = checkpoint.read_array(key).tobytes()
            reconstructed_scales = checkpoint.read_array(key + '_scale_inv').tobytes()
            if reconstructed_raw != raw or reconstructed_scales != scales:
                raise AssertionError('Reconstructed sample transport is not byte exact')
            # Independent native PyTorch FP8 cast; no AirLLM LUT in the reference.
            values = torch.from_numpy(np.frombuffer(raw, np.uint8).copy()).view(torch.float8_e4m3fn).float().reshape(shape)
            block = torch.from_numpy(np.frombuffer(scales, '<f4').copy()).reshape(tile['scale_shape'])
            reference = values * block.repeat_interleave(128, 0).repeat_interleave(128, 1)[:shape[0], :shape[1]]
            actual = np.asarray(reader.get(key))
            expected = reference.numpy()
            decode_exact = np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
            if not decode_exact:
                raise AssertionError('FP8 decoded tile differs bitwise from independent FP32 reference')
            # Fixed deterministic three-position input, independent of results.
            rng = np.random.default_rng(20261023)
            inputs = rng.normal(0, shape[1] ** -0.5, size=(3, shape[1])).astype(np.float32)
            result = reader.linear(mx.array(inputs), key)
            mx.eval(result)
            result = np.asarray(result)
            torch_result = (torch.from_numpy(inputs) @ reference.T).numpy()
            passed = np.allclose(result, torch_result, rtol=LINEAR_RTOL, atol=LINEAR_ATOL)
            if not passed:
                raise AssertionError('Quantized linear tile output exceeds predefined FP32 tolerances')
            rows.append({'key': key, 'shape': shape, 'original_rows': tile['original_rows'],
                         'source_matrix_sha256': tile['matrix_sha256'],
                         'source_scales_sha256': tile['scale_sha256'],
                         'reconstructed_source_bytes_exact': True,
                         'decoded_fp32_bitwise_equal': decode_exact,
                         'decode_max_absolute_error': float(np.max(np.abs(actual - expected))),
                         'linear_positions': 3, 'linear_rtol': LINEAR_RTOL, 'linear_atol': LINEAR_ATOL,
                         'linear_passed': bool(passed),
                         'linear_max_absolute_error': float(np.max(np.abs(result - torch_result))),
                         'linear_max_reference_absolute_value': float(np.max(np.abs(torch_result)))})
    finally:
        reader.close()
        checkpoint.close()
    return {'status': 'passed', 'scope': 'Two block-aligned real trained routed-expert matrix subranges only',
            'validated_at_utc': datetime.now(timezone.utc).isoformat(),
            'acquisition_manifest_written_at_utc': datetime.fromtimestamp((directory / 'manifest.json').stat().st_mtime, timezone.utc).isoformat(),
            'platform': platform.platform(), 'machine': platform.machine(),
            'model': MODEL, 'revision': REVISION, 'acquisition': manifest,
            'compute_device': 'MLX CPU', 'torch_version': torch.__version__, 'mlx_version': mx.__version__,
            'reference': 'PyTorch float8_e4m3fn cast and explicit official FP32 block scales',
            'matrix_rebasing': 'Original source rows begin at128-block boundaries; local row0 receives the corresponding original scale row, with no source-byte changes.',
            'tiles': rows, 'reader_stats': reader.stats, 'checkpoint_stats': checkpoint.stats,
            'throughput_benchmark_performed': False, 'full_matrix_validated': False,
            'full_expert_validated': False, 'full_shard_downloaded': False,
            'full_checkpoint_generation_validated': False,
            'metal_validated': False,
            'activation_quantization_semantics_validated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir', type=Path, default=REPO.parent / 'models/glm53-samples')
    parser.add_argument('--output', type=Path, default=REPO / 'evidence-v5/glm53-real-sample.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--acquire-only', action='store_true')
    mode.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if not args.validate_only:
        manifest = acquire(args.sample_dir)
        print(json.dumps({'acquired_body_bytes': manifest['total_response_body_bytes'],
                          'tiles': len(manifest['tiles']), 'http_range_requests': len(manifest['range_requests'])}))
    if not args.acquire_only:
        report = validate(args.sample_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'output': str(args.output), 'tiles': len(report['tiles'])}))


if __name__ == '__main__':
    main()
