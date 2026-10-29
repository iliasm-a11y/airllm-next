#!/usr/bin/env python3
"""Reproduce the offline GGML C IQ differential validation and save evidence.

Run from any directory with the test requirements installed. Model generation
and Apple M5 performance are deliberately outside this codec-only harness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'air_llm'))
sys.path.insert(0, str(ROOT / 'tests_next'))

from airllm.nextgen.gguf import GGMLType, dequantize
from test_iq_ggml_reference import (
    FIXTURES, REAL_SAMPLES, GGMLIQOracle, IQ_LAYOUTS, REVISION, iq_blocks,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = []
    special_cases = []
    with tempfile.TemporaryDirectory(prefix='iq-c-oracle-') as directory:
        oracle = GGMLIQOracle(directory)
        for name in IQ_LAYOUTS:
            kind = getattr(GGMLType, name)
            for mode in ['random', 'grids']:
                raw = iq_blocks(kind, mode=mode)
                expected = oracle.decode(raw, kind)
                actual = dequantize(raw.tobytes(), kind)
                mismatches = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
                absolute_error = float(np.max(np.abs(actual - expected)))
                cases.append({
                    'codec': name, 'fixture': mode,
                    'blocks': len(raw), 'values': int(actual.size),
                    'raw_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                    'reference_fp32_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
                    'actual_fp32_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                    'bitwise_mismatches': mismatches, 'max_abs_error': absolute_error,
                    'passed': mismatches == 0,
                })
            raw = iq_blocks(kind, count=4)
            raw[:, :2] = np.array([0x7c00, 0xfc00, 0x7e01, 0xfe01], dtype='<u2').view(np.uint8).reshape(-1, 2)
            expected = oracle.decode(raw, kind)
            with np.errstate(invalid='ignore'):
                actual = dequantize(raw.tobytes(), kind)
            special_cases.append({
                'codec': name, 'blocks': 4, 'values': int(actual.size),
                'classification_passed': bool(
                    np.array_equal(np.isnan(actual), np.isnan(expected))
                    and np.array_equal(np.isposinf(actual), np.isposinf(expected))
                    and np.array_equal(np.isneginf(actual), np.isneginf(expected))),
            })
        audit = json.loads((REAL_SAMPLES / 'checkpoint-audit-summary.json').read_text())
        for name, (type_id, block_bytes) in IQ_LAYOUTS.items():
            matching = [entry for entry in audit['samples'] if entry['type_id'] == type_id]
            if len(matching) != 1:
                raise ValueError(f'Exactly one real checkpoint sample is required for {name}')
            entry = matching[0]
            if Path(entry['file']).name != entry['file']:
                raise ValueError('A real sample must be in the audit directory')
            raw = (REAL_SAMPLES / entry['file']).read_bytes()
            if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError(f'Real checkpoint sample integrity mismatch for {name}')
            kind = getattr(GGMLType, name)
            expected = oracle.decode(raw, kind)
            actual = dequantize(raw, kind)
            mismatches = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
            cases.append({
                'codec': name, 'fixture': 'real_checkpoint_sample',
                'blocks': len(raw) // block_bytes, 'values': int(actual.size),
                'source_revision': audit['revision'], 'source': entry,
                'raw_sha256': hashlib.sha256(raw).hexdigest(),
                'reference_fp32_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
                'actual_fp32_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                'bitwise_mismatches': mismatches,
                'max_abs_error': float(np.max(np.abs(actual - expected))),
                'passed': mismatches == 0,
            })
        conversion = np.empty(65536, dtype=np.float32)
        oracle.lib.oracle_convert_all_half(conversion.ctypes.data)
        expected_conversion = np.arange(65536, dtype='<u2').view('<f2').astype(np.float32)
        finite = np.isfinite(expected_conversion)
        conversion_ok = bool(
            np.array_equal(conversion[finite].view(np.uint32), expected_conversion[finite].view(np.uint32))
            and np.array_equal(np.isnan(conversion), np.isnan(expected_conversion))
            and np.array_equal(np.isposinf(conversion), np.isposinf(expected_conversion))
            and np.array_equal(np.isneginf(conversion), np.isneginf(expected_conversion)))
        compiler = subprocess.run([oracle.command[0], '--version'], capture_output=True,
                                  text=True, check=True, timeout=10).stdout.splitlines()[0]
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'host': {'system': platform.system(), 'machine': platform.machine(),
                 'python': platform.python_version(), 'numpy': np.__version__,
                 'compiler': compiler},
        'ggml_revision': REVISION,
        'reference_provenance': json.loads((FIXTURES / 'manifest.json').read_text()),
        'compile_flags': ['-std=c11', '-O2', '-fno-fast-math', '-ffp-contract=off', '-fPIC'],
        'numeric_policy': {'finite_scales': 'bitwise FP32 equality, including signed zero',
                           'special_scales': 'NaN and signed infinity classification only',
                           'absolute_tolerance': 0, 'relative_tolerance': 0},
        'cases': cases,
        'special_scale_cases': special_cases,
        'total_blocks': sum(case['blocks'] for case in cases),
        'total_values': sum(case['values'] for case in cases),
        'fp16_conversion_patterns_checked': 65536,
        'fp16_conversion_passed': conversion_ok,
        'codec_differential_passed': (conversion_ok and all(case['passed'] for case in cases)
                                      and all(case['classification_passed'] for case in special_cases)),
        'real_checkpoint_generation_validated': False,
        'm5_execution_validated': False,
        'm5_performance_measured': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in [
        'codec_differential_passed', 'total_blocks', 'total_values',
        'fp16_conversion_passed', 'm5_performance_measured']}))
    return 0 if report['codec_differential_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
