"""One bounded positional reader across an explicit GGUF shard set.

Tensor offsets stay local to their owning file. Returned decoded arrays share
one IO budget across every shard, including arrays retained by the caller.
This is not an RSS or filesystem page-cache limit.
"""
from contextlib import contextmanager
from pathlib import Path
import re
from threading import RLock

from .gguf import GGUFError, GGUFMemoryError, GGUFReader


class SplitGGUFReader:
    def __init__(self, paths, *, memory_budget_bytes=128 << 20,
                 max_metadata_bytes=96 << 20, **kwargs):
        self.readers = []
        self.tensors = {}
        self.owners = {}
        self._lock = RLock()
        self._closed = False
        for name, value in (('memory_budget_bytes', memory_budget_bytes),
                            ('max_metadata_bytes', max_metadata_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        self.memory_budget_bytes = memory_budget_bytes
        self.max_read_bytes = kwargs.get('max_read_bytes')
        self._peak = 0
        if isinstance(paths, (str, Path)):
            first = Path(paths).expanduser()
            match = re.fullmatch(r'(.+)-00001-of-(\d{5})\.gguf', first.name)
            if match:
                count = int(match[2])
                if not 1 <= count <= 128:
                    raise GGUFError('Invalid shard count in filename')
                paths = [first.with_name(f'{match[1]}-{i:05d}-of-{count:05d}.gguf')
                         for i in range(1, count + 1)]
            else:
                paths = [first]
        self.paths = tuple(Path(p).expanduser() for p in paths)
        if not 1 <= len(self.paths) <= 128:
            raise GGUFError('Expected between 1 and 128 explicit shards')
        try:
            identities = set()
            for path in self.paths:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in identities:
                    raise GGUFError('Duplicate GGUF shard or hardlink')
                identities.add(identity)
                remaining = memory_budget_bytes - self.metadata_bytes
                remaining_metadata = max_metadata_bytes - self.metadata_bytes
                if remaining <= 0 or remaining_metadata <= 0:
                    raise GGUFMemoryError('Shard metadata exceeds shared IO budget')
                reader = GGUFReader(path, memory_budget_bytes=remaining,
                    max_metadata_bytes=min(remaining_metadata, remaining), **kwargs)
                self.readers.append(reader)
                for name, info in reader.tensors.items():
                    if name in self.tensors:
                        raise GGUFError(f'Duplicate tensor across shards: {name}')
                    self.tensors[name] = info
                    self.owners[name] = reader
            self.metadata = dict(self.readers[0].metadata)
            self.file_size = sum(r.file_size for r in self.readers)
            self._validate_splits()
        except BaseException:
            self.close()
            raise

    def _validate_splits(self):
        split_keys = {'split.count', 'split.no', 'split.tensors.count'}
        if len(self.readers) == 1 and not split_keys.intersection(self.metadata):
            return
        expected = len(self.readers)
        for index, reader in enumerate(self.readers):
            m = reader.metadata
            if any(type(m.get(key)) is not int for key in split_keys):
                raise GGUFError('GGUF split metadata requires all three integer fields')
            if m.get('split.count') != expected or m.get('split.no') != index:
                raise GGUFError('Incomplete or incorrectly ordered GGUF shard set')
            if m.get('split.tensors.count') != len(self.tensors):
                raise GGUFError('GGUF global tensor count does not match shard set')
            # The first shard is authoritative and may contain no tensors.
            # Later shards may omit model/tokenizer metadata, as the actual
            # GLM-5.3 files do, but may not silently introduce another config.
            for key in m:
                if key != 'split.no' and (key not in self.metadata or self.metadata[key] != m[key]):
                    raise GGUFError(f'Conflicting GGUF shard metadata: {key}')

    @property
    def metadata_bytes(self):
        return sum(r.metadata_bytes for r in self.readers)

    @property
    def available_memory_bytes(self):
        return max(0, self.memory_budget_bytes - self.metadata_bytes -
                   sum(r._reserved for r in self.readers))

    @contextmanager
    def _owner(self, name):
        with self._lock:
            if self._closed:
                raise ValueError('SplitGGUFReader is closed')
            reader = self.owners[name]
            # Other shards' metadata and live arrays consume the same budget.
            reader.memory_budget_bytes = self.memory_budget_bytes - sum(
                r.metadata_bytes + r._reserved for r in self.readers if r is not reader)
            try:
                yield reader
            finally:
                self._peak = max(self._peak, self.memory_budget_bytes - self.available_memory_bytes)

    def max_rows(self, name, *, extra_bytes=0):
        with self._owner(name) as reader:
            return reader.max_rows(name, extra_bytes=extra_bytes)

    def estimate_read_bytes(self, name, count):
        return self.owners[name].estimate_read_bytes(name, count)

    def read_rows(self, name, start, count, *, expert=None):
        with self._owner(name) as reader:
            return reader.read_rows(name, start, count, expert=expert)

    def read_packed_rows(self, name, start, count, **kwargs):
        with self._owner(name) as reader:
            return reader.read_packed_rows(name, start, count, **kwargs)

    def read(self, name):
        info = self.tensors[name]
        return self.read_rows(name, 0, info.rows).reshape(info.shape)

    def filenos(self):
        with self._lock:
            if self._closed:
                raise ValueError('SplitGGUFReader is closed')
            return tuple(r.fileno() for r in self.readers)

    @property
    def stats(self):
        keys = set().union(*(r.stats for r in self.readers))
        result = {k: (max(r.stats.get(k, 0) for r in self.readers) if k.startswith('peak_')
                      else sum(r.stats.get(k, 0) for r in self.readers)) for k in keys}
        result['shared_retained_peak_bytes'] = self._peak
        return result

    def close(self):
        with self._lock:
            self._closed = True
            for reader in self.readers:
                reader.close()

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise ValueError('SplitGGUFReader is closed')
            return self

    def __exit__(self, *_):
        self.close()
