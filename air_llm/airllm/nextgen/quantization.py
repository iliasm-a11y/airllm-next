"""Block FP8 reference and MLX-compatible groupwise affine 4/8-bit packing.

Quantization is opt-in and changes weights. Scale/bias are stored in float32;
quality must be evaluated for each real model, not inferred from tiny fixtures.
"""
from dataclasses import dataclass
import math
import numpy as np


def fp8_lut():
    b=np.arange(256,dtype=np.uint16);sign=np.where(b&128,-1.,1.)
    exp=(b>>3)&15;mant=b&7
    value=np.where(exp==0,np.ldexp(mant.astype(np.float32),-9),
                   np.exp2(exp.astype(np.int16)-7)*(1+mant/8))*sign
    value=np.where((exp==15)&(mant==7),np.nan,value)
    return value.astype(np.float32)

FP8_LUT=fp8_lut()


def dequant_fp8(raw,scales,block=(128,128),*,row_offset=0):
    if raw.ndim!=2:raise ValueError('Expected 2D FP8 tile')
    br,bc=block
    if br<1 or bc<1:raise ValueError('Invalid FP8 block')
    r=np.arange(raw.shape[0])+row_offset;c=np.arange(raw.shape[1])
    if scales.shape[0] <= (r[-1]//br if len(r) else -1) or scales.shape[1] != math.ceil(raw.shape[1]/bc):
        raise ValueError('Invalid FP8 scales')
    return FP8_LUT[raw]*scales[(r//br)[:,None],(c//bc)[None,:]]


@dataclass
class PackedMatrix:
    weight: object
    scales: object
    biases: object
    shape: tuple[int,int]
    bits: int
    group_size: int

    @property
    def nbytes(self):return sum(x.nbytes for x in (self.weight,self.scales,self.biases))


def quantize_affine(weight,bits=4,group_size=64):
    w=np.asarray(weight,dtype=np.float32)
    if w.ndim!=2 or bits not in (4,8) or group_size not in (32,64,128):
        raise ValueError('Affine supports matrices, bits=4/8, groups=32/64/128')
    if w.shape[1]%group_size:raise ValueError('Input columns must be divisible by group_size')
    if not np.isfinite(w).all():raise ValueError('Nonfinite weights cannot be quantized')
    n,m=w.shape;g=w.reshape(n,-1,group_size);lo=g.min(-1);hi=g.max(-1)
    scales=(hi-lo)/((1<<bits)-1);scales=np.where(scales==0,1,scales).astype(np.float32)
    q=np.clip(np.rint((g-lo[...,None])/scales[...,None]),0,(1<<bits)-1).astype(np.uint32).reshape(n,m)
    pack=32//bits;q=q.reshape(n,-1,pack)
    packed=np.bitwise_or.reduce(q << (np.arange(pack,dtype=np.uint32)*bits),axis=-1)
    return PackedMatrix(packed,scales,lo.astype(np.float32),(n,m),bits,group_size)


def dequant_affine(p):
    pack=32//p.bits
    q=((np.asarray(p.weight)[...,None] >> (np.arange(pack,dtype=np.uint32)*p.bits)) & ((1<<p.bits)-1))
    q=q.reshape(p.shape).astype(np.float32).reshape(p.shape[0],-1,p.group_size)
    return (q*np.asarray(p.scales)[...,None]+np.asarray(p.biases)[...,None]).reshape(p.shape)
