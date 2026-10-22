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
import numpy as np
from safetensors.numpy import save_file,load_file
from ..ultra.checkpoint import atomic_json
from .quantization import PackedMatrix,quantize_affine,dequant_fp8


class QuantizedSidecar:
    def __init__(self,root,fingerprint,bits=4,group_size=64,create=False):
        self.root=Path(root);self.path=self.root/'manifest.json';self.fingerprint=fingerprint
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

    def load(self,ident,rows=None):
        v=self.entries[ident];path=self.root/v['file']
        # Safetensors partial reads, including a tiled quantized output head.
        # Older manifests without cached array offsets remain readable.
        with path.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0]
            if length>1_000_000:raise ValueError('Invalid quantized header')
            header=json.loads(f.read(length));data={}
            for name in ('weight','scales','biases'):
                d=header[name];dt={'U32':'<u4','F32':'<f4'}[d['dtype']]
                shape=list(d['shape']);start,stop=(0,shape[0]) if rows is None else rows
                if not 0<=start<=stop<=shape[0]:raise ValueError('Invalid packed row tile')
                offset=8+length+d['data_offsets'][0]+start*shape[1]*4
                f.seek(offset);size=(stop-start)*shape[1]*4;raw=f.read(size)
                if len(raw)!=size:raise EOFError('Truncated quantized sidecar')
                data[name]=np.frombuffer(raw,dtype=dt).reshape(stop-start,shape[1])
        shape=(v['shape'][0] if rows is None else rows[1]-rows[0],v['shape'][1])
        return PackedMatrix(data['weight'],data['scales'],data['biases'],shape,v.get('bits',self.bits),self.group_size)

    def add(self,ident,tiles,shape,bits=None):
        if ident in self.entries:return False
        n,m=shape;bits=self.bits if bits is None else bits
        if m%self.group_size:raise ValueError(f'Unsupported column count in {ident}')
        pack=32//bits
        weight=np.empty((n,m//pack),np.uint32);scales=np.empty((n,m//self.group_size),np.float32);biases=np.empty_like(scales)
        pos=0;error_sq=0.;signal_sq=0.;max_error=0.
        from .quantization import dequant_affine
        for tile in tiles:
            q=quantize_affine(tile,bits,self.group_size);stop=pos+tile.shape[0]
            if stop>n:raise ValueError('Too many matrix rows')
            weight[pos:stop]=q.weight;scales[pos:stop]=q.scales;biases[pos:stop]=q.biases;pos=stop
            error=dequant_affine(q)-tile;error_sq+=float(np.sum(error.astype(np.float64)**2));signal_sq+=float(np.sum(tile.astype(np.float64)**2));max_error=max(max_error,float(np.max(np.abs(error))))
        if pos!=n:raise ValueError('Truncated conversion')
        filename=sha256(ident.encode()).hexdigest()+'.safetensors'
        fd,tmp=tempfile.mkstemp(prefix='.convert-',dir=self.root);os.close(fd)
        try:
            save_file({'weight':weight,'scales':scales,'biases':biases},tmp)
            with open(tmp,'rb') as f:os.fsync(f.fileno())
            os.replace(tmp,self.root/filename)
            self.entries[ident]={'file':filename,'shape':list(shape),'bits':bits,'nbytes':weight.nbytes+scales.nbytes+biases.nbytes,
              'relative_l2_error':math.sqrt(error_sq/signal_sq) if signal_sq else 0.,'max_abs_error':max_error}
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
