# SPDX-License-Identifier: Apache-2.0
"""Regenerable whole fixtures, canonical BF16 bits, independent cache encodings."""
import hashlib
import numpy as np
CASES=[('c8_4096',[4096]*8,False,False),('c33_4096',[4096]*33,False,False),
       ('c64_4096',[4096]*64,False,False),('c64_5120',[5120]*64,False,False),
       ('c8_ragged_strided',[33,65,127,512,2048,4096,5000,5120],True,False),
       ('c64_ragged_strided',[4096-(i%7)*131 for i in range(64)],True,False),
       ('c33_17408',[17408]*33,False,False),('c64_capture1_to5120',[5120]*64,False,True)]

def bits(x):
    u=np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((u+np.uint32(32767)+((u>>np.uint32(16))&np.uint32(1)))>>np.uint32(16)).astype(np.uint16)
def values(b):return (np.asarray(b,np.uint16).astype(np.uint32)<<np.uint32(16)).view(np.float32)
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def generate(seed,n,width,phase):
    rng=np.random.Generator(np.random.PCG64(seed))
    k=bits(rng.standard_normal((n,width*16,2,128),dtype=np.float32))
    v=bits(rng.standard_normal((n,width*16,2,128),dtype=np.float32))
    q=bits(rng.standard_normal((n,16,128),dtype=np.float32))
    if phase:k=bits(values(k)*np.float32(.5));v=v^np.uint16(32768);q=q^np.uint16(32768)
    table=np.random.Generator(np.random.PCG64(seed+10000+phase)).permutation(n*width).astype(np.int32).reshape(n,width)
    return k,v,q,table

def k8_encode(k):
    b=k.astype(np.int32);mag=b&32767
    rounded=((mag+7+((mag>>4)&1))>>4)-928
    return (np.where(rounded>=8,np.minimum(rounded,127),0)|((b&32768)>>8)).astype(np.uint8)
def k8_decode(k):
    b=k.astype(np.uint16);mag=b&127
    return (np.where(mag>=8,(mag<<4)+14848,0)|((b&128)<<8)).astype(np.uint16)
