"""Direct indexed safetensors access. No repartitioning or duplicated model weights.

Only headers are read at construction. Cached manifests are invalidated by index
digest, relative paths, sizes and nanosecond mtimes. Tensor reads are sorted by
file/offset; safe_open mappings are reused with a bounded, thread-safe handle LRU.
"""
from collections import OrderedDict, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from threading import RLock


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, separators=(',', ':'))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class DirectCheckpoint:
    VERSION = 1

    def __init__(self, root, cache_dir=None, max_open_files=8):
        self.root = Path(root).resolve()
        if max_open_files < 1:
            raise ValueError('max_open_files must be positive')
        self.max_open_files = max_open_files
        self.lock = RLock()
        self.handles = OrderedDict()
        self.closed = False
        self.stats = defaultdict(int)
        index_path = self.root / 'model.safetensors.index.json'
        if index_path.exists():
            index_raw = index_path.read_bytes()
            weight_map = json.loads(index_raw)['weight_map']
            files = sorted(set(weight_map.values()))
        else:
            # Single file and non-indexed sharded checkpoints are supported too.
            index_raw = b''
            weight_map = None
            files = sorted(p.name for p in self.root.glob('*.safetensors'))
        if not files:
            raise FileNotFoundError(f'No local safetensors checkpoint in {self.root}')
        identities = {}
        for name in files:
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'Invalid checkpoint path: {name}')
            p = self.root / rel
            st = p.stat()  # Missing/moved external drive fails before inference.
            identities[name] = [st.st_size, st.st_mtime_ns]
        identity = {'version': self.VERSION, 'root': str(self.root),
                    'index_sha256': sha256(index_raw).hexdigest(), 'files': identities}
        self.fingerprint = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.manifest_path = Path(cache_dir) / (self.fingerprint + '.manifest.json') if cache_dir else None
        cached = None
        if self.manifest_path and self.manifest_path.exists():
            try:
                c = json.loads(self.manifest_path.read_text())
                if c['identity'] == identity:
                    cached = c['tensors']
            except (ValueError, KeyError, OSError):
                pass
        self.stats['manifest_hit'] = int(cached is not None)
        if cached is None:
            cached = {}
            for name in files:
                p = self.root / name
                with p.open('rb') as f:
                    raw = f.read(8)
                    if len(raw) != 8:
                        raise ValueError(f'Truncated safetensors file: {p}')
                    length = struct.unpack('<Q', raw)[0]
                    if length > min(100_000_000, identities[name][0] - 8):
                        raise ValueError(f'Invalid safetensors header length: {p}')
                    header = json.loads(f.read(length))
                self.stats['header_bytes'] += 8 + length
                for key, desc in header.items():
                    if key == '__metadata__':
                        continue
                    if weight_map is not None and weight_map.get(key) != name:
                        continue
                    if key in cached:
                        raise ValueError(f'Duplicate tensor: {key}')
                    a, b = desc['data_offsets']
                    if not (0 <= a <= b <= identities[name][0] - 8 - length):
                        raise ValueError(f'Invalid offsets for {key}')
                    cached[key] = {'file': name, 'offset': 8 + length + a,
                                   'bytes': b - a, 'shape': desc['shape'], 'dtype': desc['dtype']}
            if weight_map is not None and set(cached) != set(weight_map):
                raise ValueError('Checkpoint index has missing tensor entries')
            if self.manifest_path:
                atomic_json(self.manifest_path, {'identity': identity, 'tensors': cached})
        self.tensors = cached
        self.total_bytes = sum(d['bytes'] for d in cached.values())

    def keys(self, prefix=''):
        return tuple(k for k in self.tensors if not prefix or k == prefix or k.startswith(prefix + '.'))

    def nbytes(self, keys, dtype=None):
        if dtype is None:
            return sum(self.tensors[k]['bytes'] for k in keys)
        import torch
        elem = torch.empty((), dtype=dtype).element_size()
        return sum(math.prod(self.tensors[k]['shape']) * elem for k in keys)

    def _handle(self, filename):
        # OffsetCheckpoint only uses headers/pread. Keep the optional framework
        # reader lazy so byte-exact transport research needs neither Torch nor MLX.
        from safetensors import safe_open
        if self.closed:
            raise RuntimeError('Checkpoint is closed')
        if filename in self.handles:
            self.handles.move_to_end(filename)
            return self.handles[filename]
        handle = safe_open(str(self.root / filename), framework='pt', device='cpu')
        handle.__enter__()
        self.stats['file_opens'] += 1
        self.handles[filename] = handle
        while len(self.handles) > self.max_open_files:
            _, old = self.handles.popitem(last=False)
            old.__exit__(None, None, None)
        return handle

    def read(self, keys, *, copy=True, dtype=None):
        # copy=True deliberately faults the selected pages NOW, including on prefetch
        # workers. Merely opening an mmap would not overlap I/O with compute.
        out = {}
        order = sorted(keys, key=lambda k: (self.tensors[k]['file'], self.tensors[k]['offset']))
        with self.lock:
            for key in order:
                d = self.tensors[key]
                value = self._handle(d['file']).get_tensor(key)
                target = dtype if dtype is not None and value.is_floating_point() else value.dtype
                converted = value.dtype != target
                if converted:
                    value = value.to(dtype=target)
                elif copy:
                    value = value.clone()
                out[key] = value
                self.stats['tensor_reads'] += 1
                self.stats['logical_bytes'] += d['bytes']
                if copy or converted:
                    self.stats['materialized_bytes'] += value.numel() * value.element_size()
        return out

    def read_expert_slice(self, key, expert, dtype):
        d = self.tensors[key]
        if len(d['shape']) != 3 or not 0 <= expert < d['shape'][0]:
            raise ValueError(f'Invalid expert slice {expert} of {key}')
        if d['dtype'].startswith('F8_'):
            raise NotImplementedError('Packed 3D FP8 expert slices need explicit scale-layout support')
        with self.lock:
            value = self._handle(d['file']).get_slice(key)[expert]
            value = value.to(dtype=dtype, copy=True)
            self.stats['tensor_reads'] += 1
            self.stats['logical_bytes'] += d['bytes'] // d['shape'][0]
            self.stats['materialized_bytes'] += value.numel() * value.element_size()
            return value

    def close(self):
        with self.lock:
            for handle in self.handles.values():
                handle.__exit__(None, None, None)
            self.handles.clear()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
