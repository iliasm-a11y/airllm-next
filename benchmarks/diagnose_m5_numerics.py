#!/usr/bin/env python3
"""CPU diagnosis of retained M5 miniature tensors; never imports MLX.

The original tolerances and pass/fail decisions are retained. GGML C decoding
and FP64 accumulation diagnose disagreements; they do not grant model fidelity.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'air_llm'))
from airllm.nextgen.gguf import dequantize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tensor(folder, descriptor):
    path = (folder/descriptor['file']).resolve(strict=True)
    if not path.is_relative_to(folder.resolve()) or sha(path) != descriptor['sha256']:
        raise ValueError('Retained tensor path or digest differs')
    raw = path.read_bytes()
    if len(raw) != descriptor['bytes']:
        raise ValueError('Retained tensor byte length differs')
    return np.frombuffer(raw, dtype=descriptor['dtype']).reshape(descriptor['shape'])


def errors(actual, expected, atol=1e-4, rtol=5e-5):
    diff = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    allowed = atol+rtol*np.abs(expected.astype(np.float64))
    worst = np.unravel_index(np.argmax(diff/allowed), diff.shape)
    return dict(accepted=bool(np.all(diff <= allowed)),
                violating_elements=int(np.sum(diff > allowed)),
                max_absolute_error=float(diff.max()), worst_scaled_index=list(map(int,worst)),
                worst_scaled_error=float((diff/allowed)[worst]),
                actual_at_worst=float(actual[worst]), expected_at_worst=float(expected[worst]),
                absolute_error_at_worst=float(diff[worst]), allowed_error_at_worst=float(allowed[worst]),
                atol=atol, rtol=rtol)


def diagnose(report_path):
    report = json.loads(report_path.read_text())
    spec = importlib.util.spec_from_file_location('m5_runner_cpu_oracle',ROOT/'benchmarks/validate_m5_kernels.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = ROOT/'tests_next/reference/ggml_codec_reference.c'
    with tempfile.TemporaryDirectory(prefix='airllm-m5-codec-',dir=ROOT.parent) as scratch:
        library = Path(scratch)/'codec-reference.so'
        command = ['cc','-O2','-ffp-contract=off','-dynamiclib' if sys.platform=='darwin' else '-shared',
                   '-fPIC',str(source),'-o',str(library)]
        build = subprocess.run(command,capture_output=True,text=True)
        if build.returncode:
            raise RuntimeError(build.stderr)
        lib = ctypes.CDLL(str(library))
        for name in ('dequantize_row_q4_K','dequantize_row_q6_K'):
            function = getattr(lib,name)
            function.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
            function.restype = None
        result = dict(schema='airllm-m5-numerical-diagnosis-v1',
            scope='CPU diagnosis of miniature retained GPU tensors only; no model proof or tolerance change',
            report_sha256=sha(report_path), c_decoder_sha256=sha(source),
            diagnostic_source_sha256=sha(Path(__file__)), compiler_command=command,
            compiler_exit_code=build.returncode, compiler_stdout=build.stdout,compiler_stderr=build.stderr,
            scalar_oracle_source_sha256=sha(ROOT/'benchmarks/validate_m5_kernels.py'),shapes=[])
        seen = set()
        for case in report['cases']:
            key = tuple(case[k] for k in ('kind','grouped','rows','width','batch'))
            if key in seen:
                continue
            seen.add(key)
            evidence = case['numeric_evidence']
            packed, inputs, scalar, reference = [load_tensor(report_path.parent,evidence[name])
                                                for name in ('packed','input','scalar','reference')]
            kind, grouped, rows, width, batch = key
            c_weights = np.empty((*packed.shape[:-1],width),np.float32)
            function = lib.dequantize_row_q4_K if kind == 12 else lib.dequantize_row_q6_K
            function(packed.ctypes.data,c_weights.ctypes.data,c_weights.size)
            packed_rows = packed.reshape(-1,packed.shape[-1])
            scalar_weights = module.scalar_decode(packed_rows,kind,width).reshape(c_weights.shape)
            vector_weights = dequantize(packed.tobytes(),kind).reshape(c_weights.shape)
            double = inputs.astype(np.float64) @ np.swapaxes(c_weights.astype(np.float64),-1,-2)
            native_comparison = errors(reference,scalar)
            worst = tuple(native_comparison['worst_scaled_index'])
            expert, token, row = worst if grouped else (0,*worst)
            x = inputs[expert,token] if grouped else inputs[token]
            w = c_weights[expert,row] if grouped else c_weights[row]
            terms = x.astype(np.float64)*w.astype(np.float64)
            precise = math.fsum(map(float,terms))
            candidates=[]
            for candidate in report['cases']:
                if tuple(candidate[k] for k in ('kind','grouped','rows','width','batch')) != key:
                    continue
                got=load_tensor(report_path.parent,candidate['candidate_evidence'])
                candidates.append(dict(backend=candidate['backend'],token_tile=candidate['token_tile'],
                    vs_scalar=errors(got,scalar),vs_fp64=errors(got,double),vs_reference=errors(got,reference)))
            result['shapes'].append(dict(kind=kind,grouped=grouped,rows=rows,width=width,batch=batch,
                independent_c_vs_scalar_weights_exact=bool(np.array_equal(c_weights,scalar_weights)),
                independent_c_vs_vectorized_weights_exact=bool(np.array_equal(c_weights,vector_weights)),
                reference_vs_recorded_scalar=native_comparison,reference_vs_fp64=errors(reference,double),
                recorded_scalar_vs_fp64=errors(scalar,double),
                worst_reference_comparison_terms=dict(index=list(worst),fsum_fp64=precise,
                    fp64_blas=float(double[worst]), sum_absolute_products=float(np.sum(np.abs(terms))),
                    cancellation_ratio=float(np.sum(np.abs(terms))/max(abs(precise),np.finfo(float).tiny)),
                    input_max_absolute=float(np.max(np.abs(x))),weight_max_absolute=float(np.max(np.abs(w)))),
                candidates=candidates))
        result['all_scalar_weights_match_independent_c'] = all(
            s['independent_c_vs_scalar_weights_exact'] for s in result['shapes'])
        result['all_vectorized_weights_match_independent_c'] = all(
            s['independent_c_vs_vectorized_weights_exact'] for s in result['shapes'])
        result['reference_shapes_disagreeing_with_recorded_scalar'] = sum(
            not s['reference_vs_recorded_scalar']['accepted'] for s in result['shapes'])
        result['reference_shapes_disagreeing_with_fp64'] = sum(
            not s['reference_vs_fp64']['accepted'] for s in result['shapes'])
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('report',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Refusing to overwrite diagnostic evidence')
    result=diagnose(args.report.resolve(strict=True))
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k.startswith('all_') or k.startswith('reference_shapes')}))


if __name__ == '__main__':
    main()
