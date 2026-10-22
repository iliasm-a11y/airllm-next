"""Opt-in, byte-exact block transport; no quantization or changed model kernels.

The binary index has constant-size records and is read on demand. Its size is
not multiplied by Python dictionaries for every block of a multi-TB model.
Partial overlays retain the original checkpoint. Complete standalone packs carry
their tensor layout and bounded local tokenizer/configuration assets.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from hashlib import sha256
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import struct
from threading import RLock
from time import perf_counter
import zlib

import numpy as np
from ..ultra.checkpoint import atomic_json

RECORD = struct.Struct('<QIIB7x32s')
RAW, ZLIB, SHUFFLE_ZLIB, PALETTE16 = range(4)
PACK_FORMAT = 'airllm-next-lossless'
_ASSET_NAMES = frozenset(('config.json','generation_config.json','tokenizer.json',
    'tokenizer_config.json','special_tokens_map.json','added_tokens.json',
    'tokenizer.model','vocab.json','vocab.txt','merges.txt','spiece.model',
    'sentencepiece.bpe.model','chat_template.jinja','chat_template.json'))


class NativePalette:
    """Explicitly supplied decoder library; never load code from a pack manifest."""
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.decode = self.library.palette_decode
        self.decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        self.decode.restype = ctypes.c_int

    def __call__(self, payload, raw_size):
        if raw_size % 2 or raw_size < 0:
            raise ValueError('Invalid palette length')
        output = ctypes.create_string_buffer(max(1, raw_size))
        if self.decode(payload, len(payload), output, raw_size//2):
            raise ValueError('Invalid native palette stream')
        return output.raw[:raw_size]


def palette_encode(raw):
    """Low bytes verbatim; high bytes use 15 values + an exact escape stream.

    This is a representation of arbitrary 16-bit words, including signed zero,
    infinities and NaN payloads. It makes no floating-point conversion.
    """
    if len(raw) % 2:
        raise ValueError('Palette encoding needs whole 16-bit words')
    pairs = np.frombuffer(raw, np.uint8).reshape(-1, 2)
    hi = pairs[:, 1]
    counts = np.bincount(hi, minlength=256)
    table = np.argsort(-counts, kind='stable')[:15].astype(np.uint8)
    lookup = np.full(256, 15, np.uint8)
    lookup[table] = np.arange(15, dtype=np.uint8)
    codes = lookup[hi]
    padded = np.full((len(codes) + 1) // 2 * 2, 15, np.uint8)
    padded[:len(codes)] = codes
    packed = padded[::2] | (padded[1::2] << 4)
    return table.tobytes() + pairs[:, 0].tobytes() + packed.tobytes() + hi[codes == 15].tobytes()


def palette_decode(payload, raw_size):
    if raw_size % 2:
        raise ValueError('Bad palette size')
    n = raw_size // 2
    nc = (n + 1) // 2
    if len(payload) < 15 + n + nc:
        raise ValueError('Truncated palette stream')
    a = np.frombuffer(payload, np.uint8)
    table = np.concatenate((a[:15], np.zeros(1, np.uint8)))
    packed = a[15+n:15+n+nc]
    codes = np.empty(2 * nc, np.uint8)
    codes[::2] = packed & 15
    codes[1::2] = packed >> 4
    codes = codes[:n]
    escapes = a[15+n+nc:]
    mask = codes == 15
    if int(mask.sum()) != len(escapes):
        raise ValueError('Bad palette escape count')
    out = np.empty((n, 2), np.uint8)
    out[:, 0] = a[15:15+n]
    out[:, 1] = table[codes]
    out[mask, 1] = escapes
    return out.tobytes()


def encode_block(raw, codec='auto'):
    choices = [(RAW, raw)]
    if codec not in ('auto', 'raw', 'zlib', 'shuffle-zlib', 'palette16'):
        raise ValueError('Unknown lossless codec')
    if codec in ('auto', 'zlib'):
        choices.append((ZLIB, zlib.compress(raw, 1)))
    if len(raw) % 2 == 0:
        if codec in ('auto', 'shuffle-zlib'):
            a = np.frombuffer(raw, np.uint8).reshape(-1, 2)
            choices.append((SHUFFLE_ZLIB, zlib.compress(a.T.tobytes(), 1)))
        if codec in ('auto', 'palette16'):
            choices.append((PALETTE16, palette_encode(raw)))
    # Never expand an incompressible block. Raw fallback is charged explicitly.
    return min(choices, key=lambda p: len(p[1]))


def decode_block(codec, payload, raw_size, palette_decoder=None):
    if codec == RAW:
        raw = payload
    elif codec in (ZLIB, SHUFFLE_ZLIB):
        d = zlib.decompressobj()
        raw = d.decompress(payload, raw_size + 1)
        if len(raw) != raw_size or not d.eof or d.unused_data or d.unconsumed_tail:
            raise ValueError('Invalid or oversized compressed block')
        if codec == SHUFFLE_ZLIB:
            if raw_size % 2:
                raise ValueError('Invalid shuffle size')
            raw = np.frombuffer(raw, np.uint8).reshape(2, -1).T.tobytes()
    elif codec == PALETTE16:
        raw = (palette_decoder or palette_decode)(payload, raw_size)
    else:
        raise ValueError('Unsupported lossless codec id')
    if len(raw) != raw_size:
        raise ValueError('Decoded length mismatch')
    return raw


def _read_exact(fd, length, offset):
    chunks = bytearray(length)
    pos = 0
    while pos < length:
        try:
            part = os.pread(fd, length - pos, offset + pos)
        except InterruptedError:
            continue
        if not part:
            raise EOFError('Truncated lossless pack')
        chunks[pos:pos+len(part)] = part
        pos += len(part)
    return bytes(chunks)


def _standalone_assets(source, limit):
    if type(limit) is not int or limit < 0:
        raise ValueError('max_asset_bytes must be a nonnegative integer')
    root=source.root.resolve()
    paths=[root/name for name in sorted(_ASSET_NAMES) if (root/name).exists()]
    templates=root/'chat_templates'
    if templates.is_dir():paths.extend(sorted(templates.glob('*.jinja')))
    if not (root/'config.json').is_file():
        raise FileNotFoundError('Standalone conversion requires local config.json')
    assets=[];total=0
    for path in paths:
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('Standalone asset escapes the model directory or is not a regular file')
        size=path.stat().st_size;total+=size
        if total>limit:raise MemoryError('Standalone configuration/tokenizer assets exceed max_asset_bytes')
        assets.append((path,path.relative_to(root).as_posix(),size))
    return assets


def _copy_assets(assets, destination):
    result={}
    for source,name,size in assets:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        digest=sha256();copied=0
        with source.open('rb') as incoming,target.open('xb') as outgoing:
            while chunk:=incoming.read(1024**2):
                copied+=len(chunk)
                if copied>size:raise ValueError('Standalone asset changed during conversion')
                outgoing.write(chunk);digest.update(chunk)
            if copied!=size:raise ValueError('Standalone asset changed during conversion')
            outgoing.flush();os.fsync(outgoing.fileno())
        result[name]={'bytes':size,'sha256':digest.hexdigest()}
    return result


def pack_checkpoint(source, destination, *, keys=None, block_bytes=1024**2,
                    codec='auto', max_total_bytes=256*1024**2,
                    standalone=False, max_asset_bytes=256*1024**2):
    """Stream selected source tensors once. A byte limit is mandatory by default.

    Source bytes and chunk hashes bind each reconstructed block. The source's
    fingerprint is a metadata identity, not a hash of every original weight.
    """
    if type(block_bytes) is not int or block_bytes < 2 or block_bytes % 2:
        raise ValueError('block_bytes must be a positive even integer')
    if codec not in ('auto', 'raw', 'zlib', 'shuffle-zlib', 'palette16'):
        raise ValueError('Unknown lossless codec')
    if type(standalone) is not bool:raise ValueError('standalone must be boolean')
    selected = sorted(source.tensors if keys is None else set(keys))
    total = sum(source.tensors[k]['bytes'] for k in selected)
    if type(max_total_bytes) is not int or max_total_bytes < total:
        raise MemoryError('Explicit pack byte allowance is smaller than selected tensors')
    if standalone and set(selected)!=set(source.tensors):
        raise ValueError('Standalone conversion must include every source tensor')
    assets=_standalone_assets(source,max_asset_bytes) if standalone else []
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=False)
    copied_assets=_copy_assets(assets,dest)
    entries = {}
    counts = Counter()
    with (dest/'weights.bin').open('wb') as data, (dest/'blocks.idx').open('wb') as index:
        for key in selected:
            d = source.tensors[key]
            entry = dict(dtype=d['dtype'], shape=d['shape'], raw_size=d['bytes'],
                         block_bytes=block_bytes, index_offset=index.tell(),
                         blocks=(d['bytes']+block_bytes-1)//block_bytes)
            entries[key] = entry
            for start in range(0, d['bytes'], block_bytes):
                length=min(block_bytes,d['bytes']-start)
                if isinstance(source,LosslessOverlay) and key in source.entries:
                    raw=bytes(source._read_range(key,start,length))
                else:raw=bytes(source._bytes(d['file'],d['offset']+start,length))
                kind, payload = encode_block(raw, codec)
                index.write(RECORD.pack(data.tell(), len(payload), len(raw), kind, sha256(raw).digest()))
                data.write(payload)
                counts[str(kind)] += 1
        # A committed manifest must never reference buffers still pending in
        # userspace or the kernel when conversion is interrupted.
        for stream in (data,index):
            stream.flush()
            os.fsync(stream.fileno())
    meta = dict(version=1, format=PACK_FORMAT, standalone=standalone,
                source_fingerprint=source.fingerprint, entries=entries,
                raw_bytes=total, stored_bytes=(dest/'weights.bin').stat().st_size,
                index_bytes=(dest/'blocks.idx').stat().st_size, codec_counts=dict(counts))
    if standalone:meta['assets']=copied_assets
    # An interrupted pack has no committed manifest and cannot be opened.
    atomic_json(dest/'manifest.json',meta)
    return meta


class LosslessOverlay:
    """OffsetCheckpoint-compatible optional transport overlay.

    One requested array + O(block_bytes) scratch per active reader. The caller
    must budget concurrent scratch in workspace_bytes. No block cache is hidden
    here. Full original tensors never need to be allocated to read one row.
    """
    def __init__(self, source, directory, *, max_block_bytes=4*1024**2,
                 palette_decoder=None, verify_hash=True):
        if type(max_block_bytes) is not int or max_block_bytes < 1:
            raise ValueError('max_block_bytes must be a positive integer')
        if type(verify_hash) is not bool:raise ValueError('verify_hash must be boolean')
        self.source = source
        self.directory = Path(directory).resolve()
        self.lock = RLock()
        self.palette_decoder = palette_decoder
        self.verify_hash = verify_hash
        self.transport_stats = Counter()
        self.max_block_bytes = max_block_bytes
        meta = json.loads((self.directory/'manifest.json').read_text())
        if meta.get('version') != 1 or meta.get('source_fingerprint') != source.fingerprint:
            raise ValueError('Lossless pack source identity mismatch')
        self.entries = meta['entries']
        if not isinstance(self.entries,dict):raise ValueError('Invalid lossless tensor inventory')
        index_size = (self.directory/'blocks.idx').stat().st_size
        for k, e in self.entries.items():
            if k not in source.tensors or not isinstance(e,dict):
                raise ValueError('Invalid lossless tensor inventory')
            d = source.tensors[k]
            b = e.get('block_bytes')
            if (type(b) is not int or not 0 < b <= max_block_bytes or
                e.get('raw_size') != d['bytes'] or e.get('shape') != d['shape'] or e.get('dtype') != d['dtype'] or
                type(e.get('blocks')) is not int or type(e.get('index_offset')) is not int or
                e['blocks'] != (d['bytes']+b-1)//b or e['index_offset'] < 0 or
                e['index_offset']+e['blocks']*RECORD.size > index_size):
                raise ValueError('Invalid lossless tensor layout or scratch budget')
        self.data_fd = os.open(self.directory/'weights.bin', os.O_RDONLY)
        try:
            self.index_fd = os.open(self.directory/'blocks.idx', os.O_RDONLY)
        except BaseException:
            os.close(self.data_fd)
            raise
        self.data_size = os.fstat(self.data_fd).st_size
        self.closed = False

    def __getattr__(self, name):
        return getattr(self.source, name)

    @property
    def encoded_read_bytes(self):
        """File-logical bytes, including fallback reads; never physical NAND I/O."""
        with self.lock, self.source.io_lock:
            unpacked=self.source.stats['logical_bytes']-self.transport_stats['requested_raw_bytes']
            return (self.transport_stats['payload_bytes']+self.transport_stats['index_bytes']+unpacked)

    @contextmanager
    def _descriptors(self):
        # Borrow both files atomically with respect to close(). Reading a
        # closed/reused process fd can otherwise address an unrelated file.
        with self.lock:
            if self.closed:raise RuntimeError('Lossless reader is closed')
            data_fd=os.dup(self.data_fd)
            try:index_fd=os.dup(self.index_fd)
            except BaseException:
                os.close(data_fd)
                raise
        try:yield data_fd,index_fd
        finally:
            os.close(data_fd)
            os.close(index_fd)

    def _read_range(self, key, start, size):
        with self._descriptors() as (data_fd,index_fd):
            return self._read_range_open(key,start,size,data_fd,index_fd)

    def _read_range_open(self, key, start, size, data_fd, index_fd):
        e = self.entries[key]
        if not 0 <= start <= start+size <= e['raw_size'] or size > self.source.max_read_bytes:
            raise MemoryError('Lossless selection exceeds bounded read')
        output = bytearray(size)
        b = e['block_bytes']
        for i in range(start//b, (start+size+b-1)//b if size else start//b):
            record = _read_exact(index_fd, RECORD.size, e['index_offset']+i*RECORD.size)
            with self.lock:
                self.transport_stats['index_bytes'] += RECORD.size
            offset, length, raw_size, codec, digest = RECORD.unpack(record)
            expected = min(b, e['raw_size']-i*b)
            if raw_size != expected or not 0 <= length <= b or offset+length > self.data_size:
                raise ValueError('Invalid compressed block index')
            t = perf_counter()
            payload = _read_exact(data_fd, length, offset)
            io_time = perf_counter()-t
            with self.lock:
                self.transport_stats.update(payload_bytes=length, blocks=1)
                self.transport_stats['io_seconds'] += io_time
            t = perf_counter()
            try:
                raw = decode_block(codec, payload, raw_size, self.palette_decoder)
                if self.verify_hash and sha256(raw).digest() != digest:
                    raise ValueError('Lossless block checksum mismatch')
            finally:
                with self.lock:
                    self.transport_stats['decode_and_hash_seconds'] += perf_counter()-t
            lo, hi = max(start, i*b), min(start+size, (i+1)*b)
            output[lo-start:hi-start] = raw[lo-i*b:hi-i*b]
            with self.lock:
                self.transport_stats['decoded_block_bytes'] += raw_size
        return output

    def read_array(self, key, *, rows=None, expert=None, mode='pread'):
        if key not in self.entries:
            return self.source.read_array(key, rows=rows, expert=expert, mode=mode)
        if mode not in ('pread', 'mmap'):
            raise ValueError('Unknown read mode')
        _, offset, size, dt, shape = self.source._region(key, rows=rows, expert=expert)
        raw = self._read_range(key, offset-self.source.tensors[key]['offset'], size)
        with self.lock, self.source.io_lock:
            # Keep existing logical_bytes semantics: reconstructed requested bytes.
            # Encoded transfer is exposed separately and must not be confused.
            self.source.stats['requested_bytes'] += size
            self.source.stats['logical_bytes'] += size
            self.transport_stats['requested_raw_bytes'] += size
        return np.frombuffer(raw, dt).reshape(shape)

    def read_arrays(self, requests, **kwargs):
        if set(kwargs)-{'mode','max_gap_bytes','max_batch_bytes','coalesce'}:
            raise TypeError('Unknown lossless read_arrays option')
        gap=kwargs.get('max_gap_bytes',0)
        if type(gap) is not int or gap < 0:
            raise ValueError('max_gap_bytes must be a nonnegative integer')
        if kwargs.get('mode','pread') not in ('pread','mmap'):
            raise ValueError('Unknown read mode')
        reqs = [dict(key=r) if isinstance(r, str) else r for r in requests]
        if any(not isinstance(r,dict) or 'key' not in r or set(r)-{'key','rows','expert'} for r in reqs):
            raise ValueError('Expected tensor key or key/rows/expert request')
        if not any(r['key'] in self.entries for r in reqs):
            return self.source.read_arrays(reqs, **kwargs)
        budget = kwargs.get('max_batch_bytes')
        budget = self.source.max_read_bytes if budget is None else budget
        if type(budget) is not int or budget < 1:
            raise ValueError('Invalid lossless batch byte budget')
        sizes = [self.source._region(**r)[2] for r in reqs]
        if any(size>self.source.max_read_bytes for size in sizes):
            raise MemoryError('Lossless selection exceeds bounded read')
        if sum(sizes) > budget:
            raise MemoryError('Lossless batch exceeds max_batch_bytes')
        return [self.read_array(**r, mode=kwargs.get('mode', 'pread')) for r in reqs]

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                os.close(self.data_fd)
                os.close(self.index_fd)
        self.source.close()

    def __enter__(self):return self

    def __exit__(self,*_):self.close()


class _PackedLayout:
    """Metadata-only source facade; every tensor is served from the pack."""
    def __init__(self,directory,meta,max_read_bytes):
        from .storage import _DT
        if type(max_read_bytes) is not int or max_read_bytes<1:
            raise ValueError('max_read_bytes must be a positive integer')
        self.root=directory;self.fingerprint=meta['source_fingerprint']
        self.max_read_bytes=max_read_bytes;self.io_lock=RLock();self.closed=False
        self.stats=defaultdict(int);self.tensors={}
        entries=meta.get('entries')
        if not isinstance(entries,dict) or not entries:
            raise ValueError('Standalone pack needs a complete nonempty tensor layout')
        for key,entry in entries.items():
            if not isinstance(key,str) or not isinstance(entry,dict):
                raise ValueError('Invalid standalone tensor layout')
            shape=entry.get('shape');dtype=entry.get('dtype');size=entry.get('raw_size')
            if (not isinstance(shape,list) or any(type(v) is not int or v<0 for v in shape) or
                dtype not in _DT or type(size) is not int or
                size!=math.prod(shape)*np.dtype(_DT[dtype]).itemsize):
                raise ValueError('Invalid standalone tensor byte count, dtype or shape')
            self.tensors[key]={'file':'weights.bin','offset':0,'shape':shape,'dtype':dtype,'bytes':size}
        self.total_bytes=sum(d['bytes'] for d in self.tensors.values())
        if self.total_bytes!=meta.get('raw_bytes'):
            raise ValueError('Invalid standalone total byte count')

    # Reuse the exact selector and accounting contract of the offset reader.
    # Missing keys fail before _bytes; all present keys are intercepted above.
    from .storage import OffsetCheckpoint as _OffsetCheckpoint
    _region=_OffsetCheckpoint._region
    read_array=_OffsetCheckpoint.read_array
    read_arrays=_OffsetCheckpoint.read_arrays
    del _OffsetCheckpoint

    def _bytes(self,*args,**kwargs):
        raise RuntimeError('Standalone tensor was not served by its packed index')

    def keys(self,prefix=''):
        return tuple(k for k in self.tensors if not prefix or k==prefix or k.startswith(prefix+'.'))

    def close(self):
        with self.io_lock:self.closed=True


def _verify_assets(directory,meta,max_asset_bytes):
    if type(max_asset_bytes) is not int or max_asset_bytes<0:
        raise ValueError('max_asset_bytes must be a nonnegative integer')
    assets=meta.get('assets')
    if not isinstance(assets,dict) or 'config.json' not in assets:
        raise ValueError('Standalone pack is missing its configuration asset inventory')
    total=0
    for name,entry in assets.items():
        if not isinstance(name,str) or not isinstance(entry,dict):raise ValueError('Invalid standalone asset inventory')
        path=Path(name)
        allowed=(name in _ASSET_NAMES or len(path.parts)==2 and path.parts[0]=='chat_templates' and path.suffix=='.jinja')
        if not allowed or not (directory/path).resolve().is_relative_to(directory):
            raise ValueError('Invalid standalone asset path')
        size=entry.get('bytes');digest=entry.get('sha256')
        if type(size) is not int or size<0 or not isinstance(digest,str):
            raise ValueError('Invalid standalone asset metadata')
        total+=size
        if total>max_asset_bytes:raise MemoryError('Standalone assets exceed max_asset_bytes')
        target=directory/path
        if target.stat().st_size!=size:raise ValueError(f'Standalone asset length mismatch: {name}')
        actual=sha256();seen=0
        with target.open('rb') as stream:
            while chunk:=stream.read(min(1024**2,size-seen+1)):
                seen+=len(chunk)
                if seen>size:raise ValueError(f'Standalone asset changed while reading: {name}')
                actual.update(chunk)
        if seen!=size or actual.hexdigest()!=digest:
            raise ValueError(f'Standalone asset checksum mismatch: {name}')


class LosslessCheckpoint(LosslessOverlay):
    """Complete lossless checkpoint with no dependency on original shard files.

    Tensor bytes are checked per decoded block. Config/tokenizer assets are
    bounded and hashed at startup. Model kernels and their precision are intact.
    """
    def __init__(self,directory,*,max_read_bytes=512*1024**2,
                 max_block_bytes=4*1024**2,palette_decoder=None,verify_hash=True,
                 max_asset_bytes=256*1024**2):
        directory=Path(directory).resolve()
        meta=json.loads((directory/'manifest.json').read_text())
        if meta.get('format')!=PACK_FORMAT or meta.get('standalone') is not True:
            raise ValueError('A complete standalone lossless checkpoint is required')
        for name in ('weights.bin','blocks.idx'):
            target=directory/name
            if not target.resolve().is_relative_to(directory) or not target.is_file():
                raise ValueError('Standalone payload/index must be regular files inside the pack')
        _verify_assets(directory,meta,max_asset_bytes)
        layout=_PackedLayout(directory,meta,max_read_bytes)
        super().__init__(layout,directory,max_block_bytes=max_block_bytes,
                         palette_decoder=palette_decoder,verify_hash=verify_hash)
        self.stats['standalone_pack']=1


def configured_overlay(source, config):
    """Shared by both native runtimes. All original compute paths stay intact."""
    if not config.lossless_dir:
        if config.lossless_native_library:
            raise ValueError('A native palette decoder requires lossless_dir')
        return source
    # Six block-sized temporaries is a conservative allowance for the Python,
    # ctypes and shuffle paths, separate from the demand result/IO reservation.
    block_limit = min(4*1024**2, config.workspace_bytes//(6*config.io_workers))
    decoder = NativePalette(config.lossless_native_library) if config.lossless_native_library else None
    return LosslessOverlay(source, config.lossless_dir, max_block_bytes=block_limit,
                           palette_decoder=decoder, verify_hash=config.lossless_verify_hash)


def open_checkpoint(root,config,*,cache_dir=None):
    """Open the production reader for an original checkpoint or complete pack."""
    from .storage import OffsetCheckpoint
    root=Path(root).resolve();manifest=root/'manifest.json'
    if manifest.is_file():
        meta=json.loads(manifest.read_text())
        if meta.get('format')==PACK_FORMAT and meta.get('standalone') is True:
            if config.quantized_dir:
                raise ValueError('Lossless original-weight transport cannot be combined with a quantized sidecar')
            if config.lossless_dir and Path(config.lossless_dir).resolve()!=root:
                raise ValueError('A standalone checkpoint cannot use an additional lossless overlay')
            block_limit=min(4*1024**2,config.workspace_bytes//(6*config.io_workers))
            decoder=NativePalette(config.lossless_native_library) if config.lossless_native_library else None
            return LosslessCheckpoint(root,max_read_bytes=config.max_tensor_bytes,
                max_block_bytes=block_limit,palette_decoder=decoder,verify_hash=config.lossless_verify_hash)
    source=OffsetCheckpoint(root,config.metadata_dir() if cache_dir is None else cache_dir,
                            config.max_open_files,config.max_tensor_bytes)
    try:return configured_overlay(source,config)
    except BaseException:
        source.close()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('destination')
    p.add_argument('--key', action='append', dest='keys')
    p.add_argument('--max-total-bytes', type=int, default=256*1024**2)
    p.add_argument('--block-bytes', type=int, default=1024**2)
    p.add_argument('--codec', choices=['auto','raw','zlib','shuffle-zlib','palette16'], default='auto')
    p.add_argument('--standalone',action='store_true',help='Pack all tensors plus bounded local config/tokenizer assets')
    p.add_argument('--max-asset-bytes',type=int,default=256*1024**2)
    a = p.parse_args()
    from .storage import OffsetCheckpoint
    cp = OffsetCheckpoint(a.source)
    try:
        print(json.dumps(pack_checkpoint(cp, a.destination, keys=a.keys,
            block_bytes=a.block_bytes, codec=a.codec, max_total_bytes=a.max_total_bytes,
            standalone=a.standalone,max_asset_bytes=a.max_asset_bytes), indent=2))
    finally:
        cp.close()


if __name__ == '__main__':
    main()
