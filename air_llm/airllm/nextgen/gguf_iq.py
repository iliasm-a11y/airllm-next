"""Bounded FP32 dequantization for GGML's IQ2_S, IQ3_XXS and IQ3_S.

The layouts, codebooks and operation order follow llama.cpp commit
7df8b876b91208a86a6d792e1140b3deec14535b, ggml/src/ggml-quants.c (MIT).
See NOTICE-IQ-GGML.txt for provenance and license.

Each loop gathers at most eight quantized values per block. It never expands
the codebook, sign fields or the entire packed tensor into FP32 scratch.
The caller reserves its output, conservative scratch and the fixed codebooks
before importing this module. The NumPy codebooks are immutable byte views.
"""
from __future__ import annotations

import numpy as np

from .gguf_iq_tables import (
    KMASK_IQ2XS, KSIGNS_IQ2XS, IQ2S_GRID, IQ3XXS_GRID, IQ3S_GRID,
)


def _signed_group(destination: np.ndarray, scale: np.ndarray,
                  grid: np.ndarray, signs: np.ndarray,
                  *, mask_offset: int = 0) -> None:
    """Multiply then apply signs in place, preserving GGML's signed zeros."""
    np.multiply(scale[:, None], grid, out=destination)
    mask = KMASK_IQ2XS[mask_offset:mask_offset + destination.shape[1]]
    np.negative(destination, out=destination,
                where=(signs[:, None] & mask[None, :]) != 0)


def decode_iq2_s(blocks: np.ndarray, output: np.ndarray) -> None:
    """block_iq2_s = d:f16, qs[64], qh[8], scales[8] (82 bytes)."""
    d = blocks[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    for group in range(8):
        scales = blocks[:, 74 + group]
        high = blocks[:, 66 + group].astype(np.uint16)
        for lane in range(4):
            scale = ((scales >> (4 * (lane // 2))) & 15).astype(np.float32)
            db = d * (np.float32(0.5) + scale) * np.float32(0.25)
            index = blocks[:, 2 + group * 4 + lane].astype(np.uint16)
            index |= (high << (8 - 2 * lane)) & 0x300
            grid = IQ2S_GRID[index]
            signs = blocks[:, 34 + group * 4 + lane]
            start = group * 32 + lane * 8
            _signed_group(output[:, start:start + 8], db, grid, signs)


def decode_iq3_xxs(blocks: np.ndarray, output: np.ndarray) -> None:
    """block_iq3_xxs = d:f16, qs[64], scales_and_signs[32] (98 bytes)."""
    d = blocks[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    for group in range(8):
        offset = 66 + 4 * group
        aux = blocks[:, offset:offset + 4].copy().view("<u4").reshape(-1)
        db = d * (np.float32(0.5) + (aux >> 28).astype(np.float32)) * np.float32(0.5)
        for lane in range(4):
            signs = KSIGNS_IQ2XS[(aux >> (7 * lane)) & 127]
            for half in range(2):
                index = blocks[:, 2 + group * 8 + 2 * lane + half]
                grid = IQ3XXS_GRID[index]
                start = group * 32 + lane * 8 + half * 4
                _signed_group(output[:, start:start + 4], db, grid, signs,
                              mask_offset=half * 4)


def decode_iq3_s(blocks: np.ndarray, output: np.ndarray) -> None:
    """block_iq3_s = d:f16, qs[64], qh[8], signs[32], scales[4] (110 bytes)."""
    d = blocks[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    for group in range(8):
        scale = (blocks[:, 106 + group // 2] >> (4 * (group % 2))) & 15
        db = d * (np.float32(1) + np.float32(2) * scale.astype(np.float32))
        high = blocks[:, 66 + group].astype(np.uint16)
        for lane in range(4):
            signs = blocks[:, 74 + group * 4 + lane]
            for half in range(2):
                index = blocks[:, 2 + group * 8 + 2 * lane + half].astype(np.uint16)
                index |= (high << (8 - 2 * lane - half)) & 256
                grid = IQ3S_GRID[index]
                start = group * 32 + lane * 8 + half * 4
                _signed_group(output[:, start:start + 4], db, grid, signs,
                              mask_offset=half * 4)


def decode_iq(raw: bytes, kind: int, output: np.ndarray) -> None:
    """Decode complete blocks into a caller-owned, already-budgeted FP32 array."""
    size, decoder = {
        22: (82, decode_iq2_s),
        18: (98, decode_iq3_xxs),
        21: (110, decode_iq3_s),
    }[int(kind)]
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size)
    decoder(blocks, output.reshape(-1, 256))

