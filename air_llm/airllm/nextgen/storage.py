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


def encoded_read_bytes(checkpoint):
    """Serialized file bytes requested, distinct from decoded bytes and NAND I/O."""
    encoded=getattr(checkpoint,'encoded_read_bytes',None)
    return int(encoded) if encoded is not None else int(checkpoint.stats['logical_bytes'])


class OffsetCheckpoint(DirectCheckpoint):
    def __init__(self, root, cache_dir=None, max_open_files=16, max_read_bytes=512*1024**2,
                 *, profile_io=False, profile_max_intervals=8192, profile_max_groups=512):
        if not isinstance(max_read_bytes,int) or isinstance(max_read_bytes,bool) or max_read_bytes < 1:
            raise ValueError('max_read_bytes must be a positive integer')
        if type(profile_io) is not bool:raise ValueError('profile_io must be boolean')
        super().__init__(root, cache_dir, max_open_files)
        self.fds = OrderedDict()
        self.io_lock = RLock()
        self.max_read_bytes = max_read_bytes
        self._profiler = None
        if profile_io:
            self.enable_profiler(max_intervals=profile_max_intervals,max_groups=profile_max_groups)
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

    def enable_profiler(self, *, max_intervals=8192, max_groups=512):
        """Start a fresh opt-in bounded session before scheduling concurrent IO."""
        from .io_profile import IOProfiler
        with self.io_lock:
            self._profiler=IOProfiler(max_intervals=max_intervals,max_groups=max_groups,
                                      physical_counter=physical_read_bytes)
            return self._profiler

    def io_profile(self):
        """Return a JSON-serializable snapshot, or None when profiling is off."""
        profiler=getattr(self,'_profiler',None)
        return profiler.snapshot() if profiler is not None else None

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

    def _bytes(self, filename, offset, size, mode='pread', *, _profile_regions=()):
        if mode not in ('pread','mmap'): raise ValueError('mode must be pread or mmap')
        if type(offset) is not int or type(size) is not int or offset < 0 or size < 0:
            raise ValueError('Read offset and size must be nonnegative integers')
        if size > self.max_read_bytes: raise MemoryError('Read exceeds max_tensor_bytes; use row tiles')
        t=perf_counter();actual=0;complete=False;profiler=self._profiler
        try:
            with self.descriptor(filename) as fd:
                if mode == 'mmap':
                    base = offset//mmap.ALLOCATIONGRANULARITY*mmap.ALLOCATIONGRANULARITY
                    if not size: data=b''
                    else:
                        with mmap.mmap(fd,size+offset-base,access=mmap.ACCESS_READ,offset=base) as m:
                            data=m[offset-base:offset-base+size]  # Explicit fault/copy here
                    actual=len(data)
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
                        pos+=n;actual=pos
                    data=buf
            complete=True
        finally:
            if profiler is not None:
                profiler.record_read(filename,offset,actual,perf_counter()-t,mode,
                                     _profile_regions,failed=not complete)
        with self.io_lock:
            self.stats['logical_bytes']+=size;self.stats['read_calls']+=1
            self.stats['io_seconds']+=perf_counter()-t
        return data

    def _region(self,key,*,rows=None,expert=None):
        """Validate one tensor selection without touching its data pages."""
        d=self.tensors[key]; shape=list(d['shape']);offset=d['offset']
        dt=np.dtype(_DT[d['dtype']]);item=dt.itemsize
        if expert is not None:
            if (not isinstance(expert,(int,np.integer)) or isinstance(expert,(bool,np.bool_)) or
                len(shape)!=3 or not 0<=expert<shape[0]): raise ValueError('Invalid packed expert id')
            expert=int(expert)
            stride=int(np.prod(shape[1:],dtype=object))*item;offset+=expert*stride;shape=shape[1:]
        if rows is not None:
            if len(shape)!=2: raise ValueError('Row slicing requires a matrix')
            a,b=rows
            if (any(not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) for v in (a,b)) or
                not 0<=a<=b<=shape[0]): raise ValueError('Invalid matrix row range')
            a,b=int(a),int(b)
            offset+=a*shape[1]*item;shape[0]=b-a
        size=int(np.prod(shape,dtype=object))*item
        return d['file'],offset,size,dt,tuple(shape)

    def read_array(self,key,*,rows=None,expert=None,mode='pread'):
        filename,offset,size,dt,shape=self._region(key,rows=rows,expert=expert)
        if mode not in ('pread','mmap'):raise ValueError('mode must be pread or mmap')
        if size>self.max_read_bytes:raise MemoryError('Read exceeds max_tensor_bytes; use row tiles')
        profiler=getattr(self,'_profiler',None)
        if profiler is not None:
            regions=profiler.requests([(filename,offset,size,key,expert)])
            data=self._bytes(filename,offset,size,mode,_profile_regions=regions)
        else:data=self._bytes(filename,offset,size,mode)
        with self.io_lock:self.stats['requested_bytes']+=size
        return np.frombuffer(data,dtype=dt).reshape(shape)

    def read_arrays(self,requests,*,mode='pread',max_gap_bytes=0,max_batch_bytes=None,coalesce=True):
        """Read selected regions in file order, combining adjacent offsets.

        Each request is a key or a dict with key/rows/expert. Returned arrays are
        views into byte-bounded shared read buffers, in the original request
        order. Gaps are opt-in and charged as logical IO rather than hidden.
        The total backing allocation, including gaps, is validated before IO;
        callers must consume/release one bounded batch before requesting another.
        """
        if mode not in ('pread','mmap'):raise ValueError('mode must be pread or mmap')
        if not isinstance(max_gap_bytes,int) or isinstance(max_gap_bytes,bool) or max_gap_bytes<0:
            raise ValueError('max_gap_bytes must be a nonnegative integer')
        budget=self.max_read_bytes if max_batch_bytes is None else max_batch_bytes
        if not isinstance(budget,int) or isinstance(budget,bool) or budget<1:
            raise ValueError('max_batch_bytes must be a positive integer')
        regions=[];profile_requests=[];profiler=getattr(self,'_profiler',None)
        for i,request in enumerate(requests):
            if isinstance(request,str):request={'key':request}
            if not isinstance(request,dict) or set(request)-{'key','rows','expert'}:
                raise ValueError('Expected tensor key or key/rows/expert request')
            filename,offset,size,dt,shape=self._region(**request)
            if size>self.max_read_bytes:raise MemoryError('Read exceeds max_tensor_bytes; use row tiles')
            regions.append((filename,offset,size,dt,shape,i))
            if profiler is not None:
                profile_requests.append((filename,offset,size,request['key'],request.get('expert')))
        groups=[]
        for region in sorted(regions,key=lambda r:(r[0],r[1])):
            filename,offset,size,_,_,_=region
            end=offset+size
            if (coalesce and groups and filename==groups[-1][0] and offset<=groups[-1][2]+max_gap_bytes
                and max(end,groups[-1][2])-groups[-1][1]<=self.max_read_bytes):
                groups[-1][2]=max(end,groups[-1][2]);groups[-1][3].append(region)
            else:groups.append([filename,offset,end,[region]])
        allocated=sum(end-start for _,start,end,_ in groups)
        if allocated>budget:raise MemoryError('Coalesced batch exceeds max_batch_bytes; split the request batch')
        tagged=profiler.requests(profile_requests) if profiler is not None else None
        output=[None]*len(regions)
        requested=sum(r[2] for r in regions)
        for filename,start,end,entries in groups:
            if tagged is not None:
                data=self._bytes(filename,start,end-start,mode,
                                 _profile_regions=[tagged[r[5]] for r in entries])
            else:data=self._bytes(filename,start,end-start,mode)
            for _,offset,size,dt,shape,i in entries:
                output[i]=np.frombuffer(data,dtype=dt,count=size//dt.itemsize,offset=offset-start).reshape(shape)
        with self.io_lock:
            self.stats['requested_bytes']+=requested
            self.stats['coalesced_batches']+=1
            self.stats['coalesced_groups']+=len(groups)
            # Overlapping requests can share bytes; never count sharing as a gap.
            gaps=0
            for _,start,_,entries in groups:
                covered=start
                for _,offset,size,_,_,_ in entries:
                    gaps+=max(0,offset-covered);covered=max(covered,offset+size)
            self.stats['coalesced_gap_bytes']+=gaps
            self.stats['coalesced_backing_bytes']+=allocated
        return output

    def close(self):
        with self.io_lock:
            # Close admission while holding the same lock as descriptor().
            # A reader already in flight owns a duplicate and remains valid.
            self.closed=True
            for fd in self.fds.values():os.close(fd)
            self.fds.clear()
        super().close()
