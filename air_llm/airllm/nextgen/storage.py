"""Explicit offset IO over Ultra's persistent safetensors index.

pread faults bytes on the IO worker. mmap is deliberately a separate measured
mode, never counted as device IO merely because a mapping was opened. Kernel
physical read counters are process-level, not per-file, and may be unavailable.
"""
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import errno
import mmap
import os
from pathlib import Path
from threading import RLock
from time import perf_counter
import numpy as np
from ..ultra.checkpoint import DirectCheckpoint

_DT = {'F32':'<f4','F16':'<f2','BF16':'<u2','U32':'<u4','U8':'u1',
       'F8_E4M3':'u1','F8_E4M3FN':'u1','I32':'<i4','I64':'<i8','F64':'<f8'}


def physical_read_bytes():
    """Linux actual storage bytes charged to process, incl. readahead. None on macOS."""
    try:
        lines = Path('/proc/self/io').read_text().splitlines()
        return int(next(s.split(':',1)[1] for s in lines if s.startswith('read_bytes:')))
    except (OSError, StopIteration, ValueError): return None


class OffsetCheckpoint(DirectCheckpoint):
    def __init__(self, root, cache_dir=None, max_open_files=16, max_read_bytes=512*1024**2):
        super().__init__(root, cache_dir, max_open_files)
        self.fds = OrderedDict()
        self.io_lock = RLock()
        self.max_read_bytes = max_read_bytes
        # Validate full shape/dtype byte counts even on a reused metadata manifest.
        ranges=defaultdict(list)
        for key,d in self.tensors.items():
            if d['dtype'] not in _DT: raise NotImplementedError(f'Unsupported stored dtype: {d["dtype"]}')
            if any(not isinstance(x,int) or x < 0 for x in d['shape']): raise ValueError(f'Bad shape {key}')
            expected = int(np.prod(d['shape'],dtype=object))*np.dtype(_DT[d['dtype']]).itemsize
            if expected != d['bytes']: raise ValueError(f'Invalid tensor byte count: {key}')
            if d['offset'] < 8 or d['offset']+d['bytes'] > (self.root/d['file']).stat().st_size:
                raise ValueError(f'Invalid persisted tensor offsets: {key}')
            if d['bytes']:ranges[d['file']].append((d['offset'],d['offset']+d['bytes']))
            if not (self.root/d['file']).resolve().is_relative_to(self.root):
                raise ValueError('Checkpoint symlink escapes model directory')

        for entries in ranges.values():
            intervals=sorted(entries)
            if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):
                raise ValueError('Overlapping tensor storage ranges')

    @contextmanager
    def descriptor(self, name):
        # Duplicate a borrowed descriptor under lock so LRU eviction cannot close
        # an fd while a concurrent read is in flight. Only a few live borrows.
        with self.io_lock:
            if self.closed: raise RuntimeError('Checkpoint is closed')
            if name not in self.fds:
                self.fds[name] = os.open(self.root/name, os.O_RDONLY)
                self.stats['fd_opens'] += 1
            self.fds.move_to_end(name)
            fd = os.dup(self.fds[name])
            while len(self.fds) > self.max_open_files:
                _,old = self.fds.popitem(last=False); os.close(old)
        try: yield fd
        finally: os.close(fd)

    def _bytes(self, filename, offset, size, mode='pread'):
        if size > self.max_read_bytes: raise MemoryError('Read exceeds max_tensor_bytes; use row tiles')
        t=perf_counter()
        with self.descriptor(filename) as fd:
            if mode == 'mmap':
                base = offset//mmap.ALLOCATIONGRANULARITY*mmap.ALLOCATIONGRANULARITY
                if not size: data=b''
                else:
                    with mmap.mmap(fd,size+offset-base,access=mmap.ACCESS_READ,offset=base) as m:
                        data=m[offset-base:offset-base+size]  # Explicit fault/copy here
            elif mode == 'pread':
                # preadv avoids a second giant copy; fallback pread is necessary
                # on Python/macOS builds without preadv. Short reads are retried.
                buf=bytearray(size); pos=0; attempts=0
                while pos < size:
                    try:
                        if hasattr(os,'preadv'):
                            n=os.preadv(fd,[memoryview(buf)[pos:]],offset+pos)
                        else:
                            chunk=os.pread(fd,min(size-pos,8*1024**2),offset+pos)
                            n=len(chunk);buf[pos:pos+n]=chunk
                    except OSError as e:
                        if e.errno in (errno.EINTR,errno.EAGAIN) and attempts<3:
                            attempts+=1;continue
                        raise
                    if not n: raise EOFError(f'Short checkpoint read: {filename}@{offset+pos}')
                    pos+=n
                data=buf
            else: raise ValueError('mode must be pread or mmap')
        with self.io_lock:
            self.stats['logical_bytes']+=size;self.stats['read_calls']+=1
            self.stats['io_seconds']+=perf_counter()-t
        return data

    def read_array(self,key,*,rows=None,expert=None,mode='pread'):
        d=self.tensors[key]; shape=list(d['shape']);offset=d['offset']
        dt=np.dtype(_DT[d['dtype']]);item=dt.itemsize
        if expert is not None:
            if len(shape)!=3 or not 0<=expert<shape[0]: raise ValueError('Invalid packed expert id')
            stride=int(np.prod(shape[1:],dtype=object))*item;offset+=expert*stride;shape=shape[1:]
        if rows is not None:
            if len(shape)!=2: raise ValueError('Row slicing requires a matrix')
            a,b=rows
            if not 0<=a<=b<=shape[0]: raise ValueError('Invalid matrix row range')
            offset+=a*shape[1]*item;shape[0]=b-a
        size=int(np.prod(shape,dtype=object))*item
        data=self._bytes(d['file'],offset,size,mode)
        return np.frombuffer(data,dtype=dt).reshape(shape)

    def close(self):
        with self.io_lock:
            for fd in self.fds.values():os.close(fd)
            self.fds.clear()
        super().close()
