"""Independent native ARM Q8_0 activation and dot oracle from pinned GGML."""
from __future__ import annotations

import ctypes
from pathlib import Path
import numpy as np
from .q8_compatible import validate_activation, validate_matrix


class Q8NativeOracle:
    def __init__(self, libraries):
        self.root = Path(libraries).resolve()
        self.cpu = ctypes.CDLL(str(self.root / 'libggml-cpu.dylib'))
        self.quantize = self.cpu.quantize_row_q8_0
        self.quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        self.quantize.restype = None
        self.dot = self.cpu.ggml_vec_dot_q8_0_q8_0
        self.dot.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                            ctypes.c_size_t, ctypes.c_int]
        self.dot.restype = None

    def activation(self, x):
        x = validate_activation(x, x.size)
        packed = np.empty((x.size // 32, 34), np.uint8)
        self.quantize(x.ctypes.data, packed.ctypes.data, x.size)
        return packed

    def matvec(self, packed, x):
        validate_matrix(packed, x.size, 2**63 - 1)
        activation = self.activation(x)
        output = np.empty(packed.shape[0], np.float32)
        for row in range(packed.shape[0]):
            self.dot(x.size, output[row:row+1].ctypes.data, 0, packed[row].ctypes.data,
                     0, activation.ctypes.data, 0, 1)
        return output
