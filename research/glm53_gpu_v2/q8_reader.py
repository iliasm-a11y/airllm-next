"""Bounded GGUF v3 Q8_0 row reader. Does not map or convert the full model."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import struct
import numpy as np


def read_rows(path, tensor_name, first_row, rows, *, expert=0, max_read_bytes=8 << 20):
    if any(type(v) is not int or v < 0 for v in (first_row, expert)) or type(rows) is not int or not 1 <= rows <= 154880:
        raise ValueError('invalid bounded row selection')
    if type(max_read_bytes) is not int or not 1 <= max_read_bytes <= 1 << 30:
        raise ValueError('read budget must be 1 byte..1 GiB')
    path = Path(path)
    with path.open('rb') as source:
        initial = os.fstat(source.fileno())
        size = initial.st_size
        def scalar(fmt):
            count = struct.calcsize('<'+fmt)
            if source.tell() + count > 32 << 20:
                raise ValueError('header exceeds parsing budget')
            raw = source.read(count)
            if len(raw) != count:
                raise ValueError('truncated GGUF scalar')
            return struct.unpack('<'+fmt, raw)[0]
        def string(keep=True):
            count = scalar('Q')
            if count > 16 << 20 or source.tell()+count > min(size, 32 << 20):
                raise ValueError('GGUF string exceeds parsing bounds')
            if keep:
                return source.read(count).decode('utf-8')
            source.seek(count, 1)
        formats = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
        def value(kind, keep=False, depth=0):
            if depth > 2:
                raise ValueError('nested metadata exceeds probe bound')
            if kind in formats:
                return scalar(formats[kind])
            if kind == 8:
                return string(keep)
            if kind != 9:
                raise ValueError('unsupported GGUF metadata kind')
            subtype, count = scalar('I'), scalar('Q')
            if count > 2_000_000:
                raise ValueError('metadata array exceeds parsing bound')
            if subtype in formats and not keep:
                length = count * struct.calcsize(formats[subtype])
                if source.tell()+length > min(size, 32 << 20):
                    raise ValueError('metadata array exceeds file or header budget')
                source.seek(length, 1)
            else:
                for _ in range(count):
                    value(subtype, False, depth+1)
        if source.read(4) != b'GGUF' or scalar('I') != 3:
            raise ValueError('requires little-endian GGUF v3')
        tensors, metadata = scalar('Q'), scalar('Q')
        if tensors > 100000 or metadata > 100000:
            raise ValueError('GGUF item count exceeds parsing bound')
        alignment = 32
        for _ in range(metadata):
            key, kind = string(), scalar('I')
            result = value(kind, key == 'general.alignment')
            if key == 'general.alignment':
                alignment = result
        selected = None
        for _ in range(tensors):
            name, ndim = string(), scalar('I')
            if not 1 <= ndim <= 4:
                raise ValueError('tensor dimensionality outside 1..4')
            dims = [scalar('Q') for _ in range(ndim)]
            kind, offset = scalar('I'), scalar('Q')
            if name == tensor_name:
                if selected is not None:
                    raise ValueError('duplicate selected tensor')
                selected = dims, kind, offset
        if type(alignment) is not int or alignment < 1 or alignment > 4096 or alignment & (alignment-1):
            raise ValueError('invalid GGUF alignment')
        data = (source.tell()+alignment-1)//alignment*alignment
        if selected is None:
            raise ValueError('tensor not found')
        dims, kind, offset = selected
        if kind != 8 or len(dims) not in (2, 3) or not 32 <= dims[0] <= 65536 or dims[0] % 32:
            raise ValueError('selected tensor is not bounded Q8_0 [width,rows,(experts)]')
        width, total_rows = dims[:2]
        experts = dims[2] if len(dims) == 3 else 1
        if not 1 <= total_rows <= 154880 or expert >= experts or first_row+rows > total_rows:
            raise ValueError('selected rows or expert exceed Q8_0 tensor')
        row_bytes = width//32*34
        count = rows*row_bytes
        if count > max_read_bytes:
            raise MemoryError('selected bytes exceed explicit read budget')
        absolute = data+offset+(expert*total_rows+first_row)*row_bytes
        if absolute > size or count > size-absolute:
            raise ValueError('selected Q8_0 range is outside file')
        payload = np.empty((rows, row_bytes), np.uint8)
        source.seek(absolute)
        target = memoryview(payload).cast('B')
        position = 0
        while position < count:
            read = source.readinto(target[position:])
            if not read:
                raise OSError('short Q8_0 row read')
            position += read
        after = os.fstat(source.fileno())
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity(initial) != identity(after):
            raise OSError('source changed during selected Q8_0 read')
    return payload, width, {'file': path.name, 'file_bytes': size, 'tensor': tensor_name,
        'dims': dims, 'expert': expert, 'first_row': first_row, 'rows': rows,
        'absolute_offset': absolute, 'selected_bytes': count,
        'selected_sha256': hashlib.sha256(target).hexdigest()}
