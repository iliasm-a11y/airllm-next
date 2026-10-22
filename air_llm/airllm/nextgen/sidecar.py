"""Resumable, opt-in expert-only quantized sidecar, original checkpoint intact.

One matrix is committed atomically at a time. Progress metadata references only
fully written files. Interruptions may leave an unreferenced file which is
safely overwritten on resume; no base checkpoint file is ever modified.
"""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import struct
from collections import defaultdict
import numpy as np
from ..ultra.checkpoint import atomic_json
from .quantization import PackedMatrix,quantize_affine,dequant_fp8


class QuantizedSidecar:
    def __init__(self,root,fingerprint,bits=4,group_size=64,create=False):
        self.root=Path(root);self.path=self.root/'manifest.json';self.fingerprint=fingerprint
        self.stats=defaultdict(int);self._verified={}
        if self.path.exists():
            m=json.loads(self.path.read_text())
            if m.get('version')!=1 or m.get('base_fingerprint')!=fingerprint:raise ValueError('Quantized sidecar is for a different base checkpoint')
            self.bits=m['bits'];self.group_size=m['group_size'];self.entries=m['entries']
            if create and (bits!=self.bits or group_size!=self.group_size):raise ValueError('Cannot resume with a different quantization format')
        elif create:
            if bits not in (4,8) or group_size not in (32,64,128):raise ValueError('Unsupported affine quantization')
            self.root.mkdir(parents=True,exist_ok=True);self.bits=bits;self.group_size=group_size;self.entries={};self.commit()
        else:raise FileNotFoundError('No completed sidecar manifest')
        for v in self.entries.values():
            p=self.root/v['file']
            if not p.resolve().is_relative_to(self.root.resolve()) or not p.is_file():raise ValueError('Invalid/missing sidecar file')

    def commit(self):
        atomic_json(self.path,{'version':1,'base_fingerprint':self.fingerprint,'bits':self.bits,'group_size':self.group_size,'entries':self.entries})

    def load(self,ident,rows=None,*,max_bytes=None):
        v=self.entries[ident];path=self.root/v['file']
        n,m=v['shape'];bits=v.get('bits',self.bits)
        if bits not in (4,8) or self.group_size not in (32,64,128) or m%self.group_size:
            raise ValueError('Invalid packed format metadata')
        if type(n) is not int or type(m) is not int or n<1 or m<1:
            raise ValueError('Invalid packed matrix shape')
        start,stop=(0,n) if rows is None else rows
        if type(start) is not int or type(stop) is not int or not 0<=start<=stop<=n:
            raise ValueError('Invalid packed row tile')
        shapes={'weight':[n,m//(32//bits)],'scales':[n,m//self.group_size],'biases':[n,m//self.group_size]}
        size=sum((stop-start)*shape[1]*4 for shape in shapes.values())
        if max_bytes is not None and size>max_bytes:
            raise MemoryError('Packed matrix tile exceeds max_tensor_bytes before allocation')
        with path.open('rb') as f:
            raw=f.read(8)
            if len(raw)!=8:raise ValueError('Truncated quantized header')
            length=struct.unpack('<Q',raw)[0]
            file_size=os.fstat(f.fileno()).st_size
            if not 2<=length<=min(1_000_000,file_size-8):raise ValueError('Invalid quantized header')
            header=json.loads(f.read(length));data={};intervals=[]
            for name,shape in shapes.items():
                d=header.get(name,{})
                expected_dtype='U32' if name=='weight' else 'F32'
                offsets=d.get('data_offsets',[])
                if d.get('shape')!=shape or d.get('dtype')!=expected_dtype or len(offsets)!=2:
                    raise ValueError('Invalid quantized array metadata')
                a,b=offsets
                if type(a) is not int or type(b) is not int or a<0 or b-a!=math.prod(shape)*4 or b>file_size-8-length:
                    raise ValueError('Invalid quantized data offsets')
                intervals.append((a,b))
            intervals.sort()
            payload=sum(b-a for a,b in intervals)
            if (intervals[0][0]!=0 or intervals[-1][1]!=file_size-8-length or
                any(a[1]!=b[0] for a,b in zip(intervals,intervals[1:])) or payload!=v['nbytes']):
                raise ValueError('Invalid quantized payload extent')
            for name,shape in shapes.items():
                offset=8+length+header[name]['data_offsets'][0]+start*shape[1]*4
                f.seek(offset);raw=f.read((stop-start)*shape[1]*4)
                if len(raw)!=(stop-start)*shape[1]*4:raise EOFError('Truncated quantized sidecar')
                data[name]=np.frombuffer(raw,dtype='<u4' if name=='weight' else '<f4').reshape(stop-start,shape[1])
        self.stats['payload_read_bytes']+=size
        return PackedMatrix(data['weight'],data['scales'],data['biases'],(stop-start,m),bits,self.group_size)

    def verify(self, ident=None, *, require_digest=True):
        """Bounded explicit content audit; legacy manifests have no content hash.

        A source metadata fingerprint is not a cryptographic hash of its weights.
        Verification reads are accounted separately from inference demand reads.
        """
        verified=[]
        for key in ([ident] if ident is not None else self.entries):
            entry=self.entries[key];expected=entry.get('sha256')
            if expected is None:
                if require_digest:raise ValueError(f'No content digest in legacy entry {key}; reconvert or audit separately')
                continue
            path=self.root/entry['file'];st=path.stat();signature=(st.st_size,st.st_mtime_ns,st.st_ctime_ns)
            if self._verified.get(key)!=signature:
                digest=sha256()
                with path.open('rb') as stream:
                    while chunk:=stream.read(1024*1024):
                        digest.update(chunk);self.stats['integrity_bytes']+=len(chunk)
                if digest.hexdigest()!=expected:raise ValueError(f'Quantized content checksum mismatch: {key}')
                self._verified[key]=signature
            verified.append(key)
        return verified

    def add(self,ident,tiles,shape,bits=None):
        if ident in self.entries:
            self.verify(ident,require_digest=False)
            return False
        if len(shape)!=2 or any(not isinstance(v,(int,np.integer)) or v<1 for v in shape):
            raise ValueError('Expected a nonempty matrix shape')
        n,m=map(int,shape);bits=self.bits if bits is None else bits
        if bits not in (4,8) or m%self.group_size:raise ValueError(f'Unsupported quantization shape/precision in {ident}')
        pack=32//bits
        shapes={'weight':[n,m//pack],'scales':[n,m//self.group_size],'biases':[n,m//self.group_size]}
        header={};offset=0
        for name,dimensions in shapes.items():
            size=math.prod(dimensions)*4
            header[name]={'dtype':'U32' if name=='weight' else 'F32','shape':dimensions,'data_offsets':[offset,offset+size]}
            offset+=size
        raw_header=json.dumps(header,separators=(',',':')).encode()
        raw_header+=b' '*((-len(raw_header))%8)
        data_offset=8+len(raw_header)
        filename=sha256(ident.encode()).hexdigest()+'.safetensors'
        fd,tmp=tempfile.mkstemp(prefix='.convert-',dir=self.root)
        pos=0;error_sq=0.;signal_sq=0.;max_error=0.;peak_tile_bytes=0
        from .quantization import dequant_affine
        try:
            with os.fdopen(fd,'w+b') as stream:
                stream.write(struct.pack('<Q',len(raw_header)));stream.write(raw_header)
                stream.truncate(data_offset+offset)
                for tile in tiles:
                    tile=np.asarray(tile,dtype=np.float32)
                    if tile.ndim!=2 or tile.shape[1]!=m or tile.shape[0]<1:raise ValueError('Invalid conversion tile shape')
                    stop=pos+tile.shape[0]
                    if stop>n:raise ValueError('Too many matrix rows')
                    q=quantize_affine(tile,bits,self.group_size)
                    for name in shapes:
                        value=np.asarray(getattr(q,name),dtype='<u4' if name=='weight' else '<f4')
                        stream.seek(data_offset+header[name]['data_offsets'][0]+pos*shapes[name][1]*4)
                        stream.write(memoryview(np.ascontiguousarray(value)).cast('B'))
                    error=dequant_affine(q).astype(np.float64)-tile
                    error_sq+=float(np.sum(error**2));signal_sq+=float(np.sum(tile.astype(np.float64)**2))
                    max_error=max(max_error,float(np.max(np.abs(error))))
                    peak_tile_bytes=max(peak_tile_bytes,tile.nbytes+q.nbytes+error.nbytes)
                    pos=stop
                if pos!=n:raise ValueError('Truncated conversion')
                stream.flush();os.fsync(stream.fileno());stream.seek(0);digest=sha256()
                while chunk:=stream.read(1024*1024):digest.update(chunk)
            os.replace(tmp,self.root/filename)
            self.entries[ident]={'file':filename,'shape':[n,m],'bits':bits,'nbytes':offset,
              'sha256':digest.hexdigest(),'conversion':'affine-round-to-nearest-v1',
              'relative_l2_error':math.sqrt(error_sq/signal_sq) if signal_sq else 0.,'max_abs_error':max_error,
              'peak_conversion_tile_bytes':peak_tile_bytes}
            self.commit()
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        return True


def expert_matrices(cp):
    for key,d in cp.tensors.items():
        if '.experts.' not in key or key.endswith(('_scale_inv','_scale')):continue
        if len(d['shape'])==3:
            for e in range(d['shape'][0]):yield key,e,tuple(d['shape'][1:])
        elif len(d['shape'])==2:yield key,None,tuple(d['shape'])


def convert_experts(cp,model_config,destination,bits=4,group_size=64,row_chunk=128,max_matrices=None,scope="experts",linear_bits=8):
    sidecar=QuantizedSidecar(destination,cp.fingerprint,bits,group_size,create=True)
    block=tuple(model_config.get('quantization_config',{}).get('weight_block_size',[128,128]))
    converted=0
    if scope not in ('experts','mixed'):raise ValueError('Unknown conversion scope')
    matrices=list(expert_matrices(cp))
    if scope=='mixed':
        # Router, all indexers and absorbed kv_b stay at original precision.
        # Embeddings use selected original rows; the output head can be tiled.
        for key,d in cp.tensors.items():
            if len(d['shape'])!=2 or not key.endswith('.weight'):continue
            if '.experts.' in key or any(x in key for x in ('embed_tokens','indexer.','indexers_proj','mlp.gate.','kv_b_proj')):continue
            matrices.append((key,None,tuple(d['shape'])))
    for key,e,shape in matrices:
        ident=key if e is None else f'{key}#{e}'
        if ident in sidecar.entries:
            sidecar.verify(ident,require_digest=False)
            expected=bits if '.experts.' in key else linear_bits
            if sidecar.entries[ident].get('bits',sidecar.bits)!=expected:
                raise ValueError('Cannot resume with a different per-matrix precision; choose a new output directory')
            continue
        if max_matrices is not None and converted>=max_matrices:break
        d=cp.tensors[key];scale=None
        if d['dtype'].startswith('F8_'):
            scale=cp.read_array(key+'_scale_inv',expert=e)
            if cp.tensors[key+'_scale_inv']['dtype']=='BF16':scale=(scale.astype(np.uint32)<<16).view(np.float32)
        def tiles():
            for start in range(0,shape[0],row_chunk):
                raw=cp.read_array(key,expert=e,rows=(start,min(shape[0],start+row_chunk)))
                if scale is not None:yield dequant_fp8(raw,scale,block,row_offset=start)
                elif d['dtype']=='BF16':yield (raw.astype(np.uint32)<<16).view(np.float32)
                else:yield raw.astype(np.float32)
        sidecar.add(ident,tiles(),shape,bits if '.experts.' in key else linear_bits);converted+=1
    return sidecar
