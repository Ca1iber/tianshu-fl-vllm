# SPDX-License-Identifier: Apache-2.0
"""CPU reference formats; no GPU framework imports, no checkpoint mutation."""
import math
import struct
import numpy as np

FORMATS = [('W8_channel',8,0,False),('W4_sym64',4,64,False),('W4_sym128',4,128,False),('W4_aff64',4,64,True),('W4_aff128',4,128,True)]

def bf16_from_words(words):
    return (np.asarray(words,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)

def bf16_round(value):
    value=np.asarray(value,dtype=np.float32)
    assert np.isfinite(value).all()
    u=value.view(np.uint32)
    return bf16_from_words(((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16))

def quantize(w, bits, group, affine=False):
    w=np.asarray(w,dtype=np.float32)
    assert w.ndim==2 and w.shape[1]>0 and np.isfinite(w).all()
    n,k=w.shape;g=group or k;ng=(k+g-1)//g
    a=np.zeros((n,ng*g),dtype=np.float32);a[:,:k]=w;a=a.reshape(n,ng,g)
    if affine:
        lo=np.minimum(a.min(2),np.float32(0));hi=np.maximum(a.max(2),np.float32(0))
        scale=np.maximum(hi-lo,np.float32(1.e-8))/np.float32((1<<bits)-1)
        zero=np.clip(np.rint(-lo/scale),0,(1<<bits)-1).astype(np.int16)
        raw=np.rint(a/scale[:,:,None])+zero[:,:,None]
        clipped=int(np.count_nonzero((raw<0)|(raw>((1<<bits)-1))))
        q=np.clip(raw,0,(1<<bits)-1).astype(np.int8)
    else:
        limit=(1<<(bits-1))-1
        scale=np.maximum(np.max(np.abs(a),2),np.float32(1.e-8))/np.float32(limit)
        zero=np.zeros((n,ng),dtype=np.int16)
        raw=np.rint(a/scale[:,:,None]);clipped=int(np.count_nonzero((raw< -limit)|(raw>limit)))
        q=np.clip(raw,-limit,limit).astype(np.int8)
    decoded=(q.astype(np.float32)-zero[:,:,None])*scale[:,:,None]
    return q.reshape(n,-1)[:,:k],scale,zero,bf16_round(decoded.reshape(n,-1)[:,:k]),clipped

def pack_nibbles_nn(q,affine=False):
    q=np.asarray(q)
    assert q.ndim==2 and np.issubdtype(q.dtype,np.integer)
    assert np.all((q>=0)&(q<=15)) if affine else np.all((q>=-7)&(q<=7))
    n,k=q.shape
    codes=q.astype(np.int16)+(0 if affine else 8)
    padded=np.full((n+(n%2),k),0 if affine else 8,dtype=np.uint8);padded[:n]=codes
    return np.ascontiguousarray((padded[0::2]|(padded[1::2]<<4)).T)

def unpack_nibbles_nn(packed,n,affine=False):
    p=np.asarray(packed,dtype=np.uint8);k,npairs=p.shape
    assert npairs==(n+1)//2
    q=np.empty((npairs*2,k),dtype=np.int16);q[0::2]=(p&15).T;q[1::2]=(p>>4).T
    return (q[:n]-(0 if affine else 8)).astype(np.int8)

def scalar_code(value,scale,zero,bits,affine):
    divided=struct.unpack('<f',struct.pack('<f',float(value)/float(scale)))[0]
    rounded=round(divided)+int(zero)
    lo,hi=(0,(1<<bits)-1) if affine else (-((1<<(bits-1))-1),(1<<(bits-1))-1)
    return min(hi,max(lo,rounded))

def check_formats():
    samples=0
    for n,k in [(1,1),(3,259),(17,128),(130,257)]:
        x=np.arange(n*k,dtype=np.int32).reshape(n,k)
        for affine in (False,True):
            q=(x%16 if affine else x%15-7).astype(np.int8)
            packed=pack_nibbles_nn(q,affine)
            assert np.array_equal(unpack_nibbles_nn(packed,n,affine),q)
            samples+=q.size
    words=np.arange(65536,dtype=np.uint16);f=bf16_from_words(words);finite=np.isfinite(f)
    assert np.array_equal(bf16_round(f[finite]).view(np.uint32),f[finite].view(np.uint32))
    a=bf16_round(np.sin(np.arange(3*259,dtype=np.float32)).reshape(3,259));a[0]=0
    checks=0
    for name,bits,g,aff in FORMATS:
        q,s,z,d,clipped=quantize(a,bits,g,aff)
        group=g or a.shape[1]
        for row in range(a.shape[0]):
            for col in range(a.shape[1]):
                assert int(q[row,col])==scalar_code(a[row,col],s[row,col//group],z[row,col//group],bits,aff)
                checks+=1
        assert np.all(d[0]==0)
        if bits==4: assert np.array_equal(unpack_nibbles_nn(pack_nibbles_nn(q,aff),q.shape[0],aff),q)
    for v in (np.nan,np.inf,-np.inf):
        try:quantize(np.array([[v]],dtype=np.float32),4,128)
        except AssertionError:pass
        else:raise AssertionError('nonfinite input accepted')
    return dict(nibble_roundtrip_elements=samples,BF16_finite_words=int(finite.sum()),independent_scalar_codes=checks,nonfinite_rejected=True)
