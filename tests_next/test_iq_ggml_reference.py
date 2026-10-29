"""IQ2_S/IQ3_XXS/IQ3_S against unmodified, pinned GGML C decoders.

Finite-scale results must agree bit for bit, including signed zero. Special
scales are compared by classification. No production decoder/table constructs
the C oracle, and no model checkpoint or network access is needed.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from airllm.nextgen.gguf import (
    GGMLType, GGUFError, GGUFMemoryError, GGUFReader, dequantize,
)
from test_gguf import _fixture


FIXTURES = Path(__file__).parent / 'fixtures' / 'iq_ggml'
REAL_SAMPLES = Path(__file__).parents[1] / 'research' / 'ronin_aj_iq3xxs'
REVISION = '7df8b876b91208a86a6d792e1140b3deec14535b'
# ggml_type and block structures from that revision; not production _LAYOUT.
IQ_LAYOUTS = {'IQ2_S': (22, 82), 'IQ3_XXS': (18, 98), 'IQ3_S': (21, 110)}
FINITE_SCALES = np.array([
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400,
    0x8400, 0x3555, 0xb555, 0x3c00, 0xbc00, 0x7bff, 0xfbff,
], dtype='<u2')


class GGMLIQOracle:
    """Small independently compiled C reference usable by other tests."""

    def __init__(self, directory):
        if sys.byteorder != 'little':
            raise RuntimeError('The upstream C lookup tables require little endian')
        compiler = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
        if compiler is None:
            raise FileNotFoundError('A C11 compiler is required for the GGML oracle')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / ('iq-oracle.dylib' if sys.platform == 'darwin' else 'iq-oracle.so')
        command = [compiler, '-std=c11', '-O2', '-fno-fast-math',
                   '-ffp-contract=off', '-fPIC',
                   '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                   str(FIXTURES / 'oracle.c'), '-o', str(output)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        self.command = command
        self.lib = ctypes.CDLL(str(output))
        self.lib.oracle_block_bytes.argtypes = [ctypes.c_int]
        self.lib.oracle_block_bytes.restype = ctypes.c_size_t
        self.lib.oracle_decode_iq.argtypes = [ctypes.c_int, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.oracle_decode_iq.restype = ctypes.c_int
        self.lib.oracle_convert_all_half.argtypes = [ctypes.c_void_p]
        self.lib.oracle_convert_all_half.restype = None

    def decode(self, raw, kind):
        raw = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, bytes) else np.asarray(raw, dtype=np.uint8)
        raw = np.ascontiguousarray(raw)
        type_id, block_bytes = IQ_LAYOUTS[kind.name if hasattr(kind, 'name') else kind]
        if raw.size % block_bytes:
            raise ValueError('Truncated reference input')
        output = np.empty(raw.size // block_bytes * 256, dtype=np.float32)
        status = self.lib.oracle_decode_iq(type_id, raw.ctypes.data,
            raw.nbytes, output.ctypes.data, output.size)
        if status != 0:
            raise RuntimeError(f'GGML reference decode failed: {status}')
        return output


@pytest.fixture(scope='session')
def iq_ggml_oracle(tmp_path_factory):
    if sys.byteorder != 'little':
        pytest.skip('The unmodified upstream C tables require little endian')
    try:
        return GGMLIQOracle(tmp_path_factory.mktemp('iq-ggml-oracle'))
    except FileNotFoundError as error:
        pytest.skip(str(error))


def iq_blocks(kind, *, mode='random', count=4096):
    """Valid packed blocks independent of the implementation under test.

    All bit fields in these codecs are total; only the FP16 global scale is
    restricted to finite values for exact-parity tests.
    """
    name = kind.name if hasattr(kind, 'name') else kind
    _, size = IQ_LAYOUTS[name]
    if mode == 'grids':
        count = {'IQ2_S': 1024, 'IQ3_XXS': 256, 'IQ3_S': 512}[name]
    rng = np.random.default_rng(20261029 + IQ_LAYOUTS[name][0])
    raw = rng.integers(0, 256, (count, size), dtype=np.uint8)
    scales = np.resize(FINITE_SCALES, count)
    raw[:, :2] = scales.view(np.uint8).reshape(-1, 2)
    if mode == 'random':
        return raw
    if mode != 'grids':
        raise ValueError(mode)
    # A nonzero finite scale makes each sign and grid position observable.
    raw[:, :2] = np.array([0x00, 0x30], dtype=np.uint8)  # binary16 0.125
    for index, row in enumerate(raw):
        if name == 'IQ2_S':
            row[2:34] = index & 255
            row[34:66] = index & 255  # each sign byte spans all 256 masks
            row[66:74] = (index >> 8) * 0x55
            row[74:82] = (index & 15) | (((index >> 4) & 15) << 4)
        elif name == 'IQ3_XXS':
            row[2:66] = index
            signs = index & 127
            auxiliary = ((index >> 4) << 28) | sum(signs << (7 * lane) for lane in range(4))
            row[66:98] = np.tile(np.frombuffer(auxiliary.to_bytes(4, 'little'), dtype=np.uint8), 8)
        else:
            row[2:66] = index & 255
            row[66:74] = 255 if index & 256 else 0
            row[74:106] = index & 255
            row[106:110] = (index & 15) | (((index >> 4) & 15) << 4)
    return raw


def test_oracle_source_provenance_and_layout(iq_ggml_oracle):
    manifest = json.loads((FIXTURES / 'manifest.json').read_text())
    assert manifest['revision'] == REVISION
    for name, digest in manifest['vendored_sha256'].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest
    for name, (type_id, block_bytes) in IQ_LAYOUTS.items():
        assert int(getattr(GGMLType, name)) == type_id
        assert iq_ggml_oracle.lib.oracle_block_bytes(type_id) == block_bytes
    assert iq_ggml_oracle.lib.oracle_block_bytes(9999) == 0


def test_c_shim_fp16_conversion_exhaustive(iq_ggml_oracle):
    result = np.empty(65536, dtype=np.float32)
    iq_ggml_oracle.lib.oracle_convert_all_half(result.ctypes.data)
    half = np.arange(65536, dtype='<u2').view('<f2')
    expected = half.astype(np.float32)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(result[finite].view(np.uint32), expected[finite].view(np.uint32))
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(result), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(result), np.isneginf(expected))


@pytest.mark.parametrize('name', list(IQ_LAYOUTS))
@pytest.mark.parametrize('mode', ['random', 'grids'])
def test_iq_bitwise_ggml_c_parity(iq_ggml_oracle, name, mode):
    kind = getattr(GGMLType, name)
    raw = iq_blocks(kind, mode=mode)
    expected = iq_ggml_oracle.decode(raw, kind)
    actual = dequantize(raw.tobytes(), kind)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize('name', list(IQ_LAYOUTS))
def test_iq_nonfinite_scale_classification(iq_ggml_oracle, name):
    kind = getattr(GGMLType, name)
    raw = iq_blocks(kind, count=4)
    raw[:, :2] = np.array([0x7c00, 0xfc00, 0x7e01, 0xfe01], dtype='<u2').view(np.uint8).reshape(-1, 2)
    expected = iq_ggml_oracle.decode(raw, kind)
    with np.errstate(invalid='ignore'):
        actual = dequantize(raw.tobytes(), kind)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))


@pytest.mark.parametrize('name', list(IQ_LAYOUTS))
def test_real_aj_checkpoint_blocks_match_ggml_c(iq_ggml_oracle, name):
    audit = json.loads((REAL_SAMPLES / 'checkpoint-audit-summary.json').read_text())
    kind = getattr(GGMLType, name)
    entries = [entry for entry in audit['samples'] if entry['type_id'] == int(kind)]
    assert len(entries) == 1
    entry = entries[0]
    assert len(audit['revision']) == 40
    assert Path(entry['file']).name == entry['file']
    assert entry['absolute_offset'] >= 0
    raw = (REAL_SAMPLES / entry['file']).read_bytes()
    assert len(raw) == entry['bytes']
    assert hashlib.sha256(raw).hexdigest() == entry['sha256']
    expected = iq_ggml_oracle.decode(raw, kind)
    actual = dequantize(raw, kind)
    assert actual.size == 8 * 256
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize('name', list(IQ_LAYOUTS))
def test_iq_selective_expert_rows_match_c(iq_ggml_oracle, tmp_path, name):
    kind = getattr(GGMLType, name)
    raw = iq_blocks(kind, count=24)
    path = _fixture(tmp_path / 'iq-experts.gguf', [('w', (512, 3, 4), kind, raw.tobytes())])
    with GGUFReader(path) as reader:
        assert reader.stats['tensor_bytes'] == 0
        result = reader.read_rows('w', 1, 2, expert=2)
        expected = iq_ggml_oracle.decode(raw[14:18], kind).reshape(2, 512)
        np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))
        assert reader.stats['tensor_bytes'] == 4 * IQ_LAYOUTS[name][1]
        packed = reader.read_packed_rows('w', 1, 2, expert=2, copy=False)
        assert packed.tobytes() == raw[14:18].tobytes()


@pytest.mark.parametrize('name', list(IQ_LAYOUTS))
def test_iq_malformed_blocks_and_preallocation_guard(tmp_path, monkeypatch, name):
    kind = getattr(GGMLType, name)
    raw = iq_blocks(kind, count=2)
    with pytest.raises(GGUFError, match='Truncated'):
        dequantize(raw.tobytes()[:-1], kind)
    with pytest.raises(GGUFMemoryError):
        dequantize(raw.tobytes(), kind, memory_budget_bytes=1)
    assert dequantize(b'', kind).size == 0
    path = _fixture(tmp_path / 'iq-budget.gguf', [('w', (256, 2), kind, raw.tobytes())])
    # Admit the IQ codebooks/index, but not one decoded row plus scratch.
    with GGUFReader(path, memory_budget_bytes=24 << 10) as reader:
        before = dict(reader.stats)
        original = reader._pread
        def no_payload(count, offset, *, tensor=False):
            assert not tensor, 'IQ payload read happened before budget admission'
            return original(count, offset, tensor=tensor)
        monkeypatch.setattr(reader, '_pread', no_payload)
        with pytest.raises(GGUFMemoryError):
            reader.read_rows('w', 0, 1)
        assert reader.stats == before


def test_iq_unknown_type_remains_explicitly_rejected():
    with pytest.raises(GGUFError, match='Unsupported'):
        dequantize(bytes(82), 9999)
