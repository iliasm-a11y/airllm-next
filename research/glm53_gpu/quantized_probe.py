"""Bounded real-byte Q2_K microbenchmark, not a GLM decode benchmark.

Run as ``python -m research.glm53_gpu.quantized_probe --help`` from repo root.
The CPU oracle is the selected pinned llama.cpp library, never Python dot code.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import struct
import time
import numpy as np
from .quantized import Q2KMetalMatvec, dequantize_reference


class NativeOracle:
    def __init__(self, library_directory):
        root = Path(library_directory).resolve()
        self.base = ctypes.CDLL(str(root / 'libggml-base.dylib'))
        self.cpu = ctypes.CDLL(str(root / 'libggml-cpu.dylib'))
        self.quantize = self.base.quantize_row_q8_K_ref
        self.quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        self.quantize.restype = None
        self.decode = self.base.dequantize_row_q2_K
        self.decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        self.decode.restype = None
        self.dot = self.cpu.ggml_vec_dot_q2_K_q8_K
        self.dot.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                             ctypes.c_size_t, ctypes.c_int]
        self.dot.restype = None

    def activation(self, x):
        if (not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.ndim != 1
                or not x.flags.c_contiguous or x.size < 256 or x.size > 65536 or x.size % 256
                or not np.isfinite(x).all()):
            raise ValueError('native activation requires bounded contiguous finite float32 [width]')
        width = x.size
        raw = np.zeros((width // 256, 292), np.uint8)
        self.quantize(x.ctypes.data, raw.ctypes.data, width)
        scales = raw[:, :4].copy().view('<f4').reshape(-1)
        qs = raw[:, 4:260].copy().view('i1').reshape(-1)
        return raw, qs, scales

    def matvec(self, packed, x):
        raw, _, _ = self.activation(x)
        output = np.empty(packed.shape[0], np.float32)
        for row in range(packed.shape[0]):
            self.dot(x.size, output[row:row+1].ctypes.data, 0,
                     packed[row].ctypes.data, 0, raw.ctypes.data, 0, 1)
        return output

    def dequantize(self, packed, width):
        output = np.empty((packed.shape[0], width), np.float32)
        for row in range(packed.shape[0]):
            self.decode(packed[row].ctypes.data, output[row].ctypes.data, width)
        return output



class NativeMatrixOracle:
    """Native whole-matrix adapter; pools persist between measurements."""
    def __init__(self, path, native, rows, width, threads):
        self.library = ctypes.CDLL(str(Path(path).resolve()))
        self.library.airllm_q2k_oracle_create.argtypes = [ctypes.c_int] * 3
        self.library.airllm_q2k_oracle_create.restype = ctypes.c_void_p
        self.library.airllm_q2k_oracle_run.argtypes = [ctypes.c_void_p] * 4
        self.library.airllm_q2k_oracle_run.restype = ctypes.c_int
        self.library.airllm_q2k_oracle_destroy.argtypes = [ctypes.c_void_p]
        self.library.airllm_q2k_oracle_destroy.restype = None
        self.handle = self.library.airllm_q2k_oracle_create(rows, width, threads)
        if not self.handle:
            raise RuntimeError('native matrix oracle pool creation failed')
        self.native, self.rows, self.width = native, rows, width

    def matvec(self, packed, x):
        if not self.handle:
            raise RuntimeError('native matrix oracle is closed')
        if (not isinstance(packed, np.ndarray) or packed.dtype != np.uint8
                or not packed.flags.c_contiguous
                or packed.shape != (self.rows, self.width // 256 * 84)
                or not isinstance(x, np.ndarray) or x.shape != (self.width,)):
            raise ValueError('native matrix oracle input shape/dtype mismatch')
        raw, _, _ = self.native.activation(x)
        output = np.empty(self.rows, np.float32)
        result = self.library.airllm_q2k_oracle_run(self.handle, packed.ctypes.data,
                                                  raw.ctypes.data, output.ctypes.data)
        if result:
            raise RuntimeError('native matrix oracle rejected its call')
        return output

    def close(self):
        if self.handle:
            self.library.airllm_q2k_oracle_destroy(self.handle)
            self.handle = None


def read_selected_rows(path, name, expert, first, count, *, max_read_bytes=8 << 20):
    """Parse bounded GGUF v3 headers, then pread only the requested Q2_K rows."""
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as f:
        def scalar(fmt):
            n = struct.calcsize('<' + fmt)
            if f.tell() + n > (32 << 20):
                raise ValueError('header exceeds probe parsing budget')
            raw = f.read(n)
            if len(raw) != n:
                raise ValueError('truncated GGUF header')
            return struct.unpack('<' + fmt, raw)[0]
        def string(keep=True):
            n = scalar('Q')
            if n > (16 << 20) or n > size - f.tell() or f.tell() + n > (32 << 20):
                raise ValueError('oversized/truncated GGUF string')
            if keep:
                return f.read(n).decode('utf-8')
            f.seek(n, 1)
        formats = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
        def value(kind, keep=False, depth=0):
            if depth > 2:
                raise ValueError('nested metadata exceeds probe bound')
            if kind in formats:
                return scalar(formats[kind])
            if kind == 8:
                return string(keep)
            if kind == 9:
                subtype, n = scalar('I'), scalar('Q')
                if n > 2_000_000:
                    raise ValueError('metadata array exceeds probe bound')
                if subtype in formats and not keep:
                    delta = n * struct.calcsize(formats[subtype])
                    if delta > size - f.tell() or f.tell() + delta > (32 << 20):
                        raise ValueError('truncated metadata array')
                    f.seek(delta, 1)
                else:
                    for _ in range(n):
                        value(subtype, False, depth+1)
                return None
            raise ValueError(f'unsupported metadata type {kind}')
        if f.read(4) != b'GGUF' or scalar('I') != 3:
            raise ValueError('probe requires little-endian GGUF v3')
        nt, nk = scalar('Q'), scalar('Q')
        if nt > 100_000 or nk > 100_000:
            raise ValueError('GGUF item count exceeds probe bound')
        alignment = 32
        for _ in range(nk):
            key, kind = string(), scalar('I')
            v = value(kind, key == 'general.alignment')
            if key == 'general.alignment':
                alignment = v
        selected = None
        for _ in range(nt):
            tensor_name, nd = string(), scalar('I')
            if not 1 <= nd <= 4:
                raise ValueError('invalid tensor dimensionality')
            dims = [scalar('Q') for _ in range(nd)]
            kind, offset = scalar('I'), scalar('Q')
            if tensor_name == name:
                if selected is not None:
                    raise ValueError('duplicate selected tensor name')
                selected = dims, kind, offset
        if type(alignment) is not int or not 1 <= alignment <= 4096 or alignment & (alignment-1):
            raise ValueError('invalid GGUF alignment')
        data = (f.tell() + alignment - 1) // alignment * alignment
        if selected is None:
            raise ValueError('selected tensor absent')
        dims, kind, offset = selected
        if kind != 10 or len(dims) != 3 or dims[0] % 256 or dims[0] < 256:
            raise ValueError('selection must be a Q2_K expert tensor')
        width, rows, experts = dims
        if not 0 <= expert < experts or first < 0 or count < 1 or first + count > rows:
            raise ValueError('selected expert/row range outside tensor')
        row_bytes = width // 256 * 84
        nbytes = count * row_bytes
        if nbytes > max_read_bytes:
            raise MemoryError('selected read exceeds probe budget')
        absolute = data + offset + (expert * rows + first) * row_bytes
        if absolute > size or nbytes > size - absolute:
            raise ValueError('selection outside GGUF file')
        raw = os.pread(f.fileno(), nbytes, absolute)
        if len(raw) != nbytes:
            raise ValueError('short selected read')
    metadata = {'file': path.name, 'file_bytes': size, 'tensor': name, 'dims': dims,
                'expert': expert, 'first_row': first, 'rows': count,
                'absolute_offset': absolute, 'selected_bytes': nbytes,
                'sha256_selected_bytes': hashlib.sha256(raw).hexdigest()}
    return np.frombuffer(raw, np.uint8).reshape(count, row_bytes).copy(), width, metadata


def compare(a, b):
    if a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError('comparison requires equal float32 shapes')
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    error = a.astype(np.float64) - b.astype(np.float64)
    return {'finite': finite,
            'bit_identical': finite and bool(np.array_equal(a.view('u4'), b.view('u4'))),
            'max_abs': float(np.max(np.abs(error))) if finite else None,
            'rmse': float(np.sqrt(np.mean(error*error))) if finite else None,
            'elements': int(a.size)}


def evaluate_acceptance(result):
    """Fail closed if any CPU-compatible operator check is missing or fails."""
    reports = [result.get(key, {}) for key in ('q8_scales', 'decode_vs_native',
                                               'cpu_compatible_vs_native_arm_dot')]
    reports += list(result.get('native_matrix_validation', {}).values())
    return bool(result.get('q8_values_exact') is True and all(
        report.get('finite') is True and report.get('bit_identical') is True
        for report in reports))


def timed(fn, repetitions):
    values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return {'seconds': values, 'median_seconds': statistics.median(values)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True)
    parser.add_argument('--libraries', required=True)
    parser.add_argument('--cpu-helper', help='optional compiled quantized_native_oracle.cpp dylib')
    parser.add_argument('--tensor', default='blk.3.ffn_gate_exps.weight')
    parser.add_argument('--expert', type=int, default=0)
    parser.add_argument('--first-row', type=int, default=0)
    parser.add_argument('--rows', type=int, default=32)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.rows <= 8192 or not 1 <= args.repetitions <= 20:
        parser.error('rows must be 1..8192; repetitions must be 1..20')
    import mlx.core as mx
    mx.set_cache_limit(0)
    mx.set_memory_limit(128 << 20)  # MLX guideline; payload guard remains mandatory.
    packed, width, metadata = read_selected_rows(args.gguf, args.tensor, args.expert,
                                                args.first_row, args.rows)
    rng = np.random.default_rng(5305)
    x = rng.normal(size=width).astype('f4')
    oracle = NativeOracle(args.libraries)
    matrix = Q2KMetalMatvec(packed, width)
    native = oracle.matvec(packed, x)
    q8, scales = matrix.quantize_activations(x)
    mx.eval(q8, scales)
    _, cpu_q8, cpu_scales = oracle.activation(x)
    gpu_q8 = np.asarray(q8)
    gpu_scales = np.asarray(scales)
    f32 = np.asarray(matrix(x, mode='float'))
    compatible = np.asarray(matrix(x, mode='cpu_q8k'))
    validation_rows = min(packed.shape[0], 32)
    native_decoded = oracle.dequantize(packed[:validation_rows], width)
    literal = dequantize_reference(packed[:validation_rows], width)
    # Independent native decode and CPU BLAS reference, deliberately timed apart.
    native_float = native_decoded @ x
    matrix_oracles = ({n: NativeMatrixOracle(args.cpu_helper, oracle, packed.shape[0], width, n)
                       for n in (1, 10)} if args.cpu_helper else {})
    helper_validation = {str(n): compare(value.matvec(packed, x), native)
                         for n, value in matrix_oracles.items()}
    result = {
        'schema': 1, 'scope': 'selected Q2_K rows; seeded synthetic activation; not GLM decode',
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'metadata': metadata, 'mlx_version': mx.__version__, 'activation_seed': 5305,
        'q8_values_exact': bool(np.array_equal(gpu_q8, cpu_q8)),
        'q8_scales': compare(gpu_scales, cpu_scales),
        'decode_vs_native': compare(literal, native_decoded),
        'cpu_compatible_vs_native_arm_dot': compare(compatible, native),
        'float_vs_native_arm_dot': compare(f32, native),
        'float_vs_native_dequant_blas': compare(f32[:validation_rows], native_float),
        'native_dequant_validation_rows': validation_rows,
        'native_libraries_sha256': {name: hashlib.sha256((Path(args.libraries) / name).read_bytes()).hexdigest()
                                  for name in ('libggml-base.dylib', 'libggml-cpu.dylib')},
        'known_payload_bytes': matrix.payload_bytes,
        'mlx_peak_bytes': mx.get_peak_memory(),
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'timing_limits': ['warm prepared matrix', 'MLX calls include activation import and synchronization',
                          'native CPU calls are one dot per row through ctypes, one caller thread',
                          'other application/compilation workload and macOS file cache are uncontrolled',
                          'not CPU10 whole-model latency; no end-to-end speedup inferred'],
        'native_cpu_dot': timed(lambda: oracle.matvec(packed, x), args.repetitions),
        'gpu_cpu_compatible': timed(lambda: matrix(x, mode='cpu_q8k'), args.repetitions),
        'gpu_float': timed(lambda: matrix(x, mode='float'), args.repetitions),
    }
    if matrix_oracles:
        result['native_matrix_helper_sha256'] = hashlib.sha256(Path(args.cpu_helper).read_bytes()).hexdigest()
        result['native_matrix_validation'] = helper_validation
        result['native_matrix_timing_limits'] = [
            'one ctypes call dispatches whole matrix; no per-row Python overhead',
            'Q8_K activation quantization included; native dot arithmetic unchanged',
            'CPU10 uses persistent10-worker pool; creation excluded, dispatch included',
            'component only; different scheduler and operation shape from whole-model decode']
        for n, value in matrix_oracles.items():
            result[f'native_matrix_cpu{n}'] = timed(lambda: value.matvec(packed, x), args.repetitions)
            value.close()
    result['mlx_peak_bytes'] = mx.get_peak_memory()
    result['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['acceptance'] = {'criterion': 'finite bit-identical Q8_K activations, original Q2_K decode and CPU-compatible ARM dot',
                            'passed': evaluate_acceptance(result)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['q8_values_exact', 'q8_scales',
        'cpu_compatible_vs_native_arm_dot', 'float_vs_native_arm_dot',
        'known_payload_bytes', 'mlx_peak_bytes', 'process_peak_rss_bytes']}, indent=2))
    return 0 if result['acceptance']['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
