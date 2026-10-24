# SPDX-License-Identifier: Apache-2.0
"""Finite BF16-input CPU contract; this module does not implement a GPU kernel."""
import numpy as np
MIN_AMPLITUDE=np.float32(2.0**-100)
PAGE_BYTES=10496
V_OFFSET=2304
SCALE_OFFSET=2048

def pack(q):
    q=np.asarray(q)
    assert q.shape[-1]==128 and np.all(q>=-8) and np.all(q<=7)
    u=(q.astype(np.int16)+8).astype(np.uint8)
    return (u[...,0::2] | (u[...,1::2]<<np.uint8(4))).copy()

def unpack(packed):
    packed=np.asarray(packed,dtype=np.uint8)
    assert packed.shape[-1]==64
    q=np.empty((*packed.shape[:-1],128),dtype=np.int8)
    q[...,0::2]=(packed & np.uint8(15)).astype(np.int16)-8
    q[...,1::2]=(packed >> np.uint8(4)).astype(np.int16)-8
    return q

def encode(x):
    x=np.asarray(x,dtype=np.float32)
    assert x.shape[-1]==128 and np.isfinite(x).all()
    maximum=np.max(np.abs(x),axis=-1)
    scale=np.where(maximum<MIN_AMPLITUDE,np.float32(1),maximum/np.float32(7)).astype(np.float32)
    inverse=(np.float32(1)/scale).astype(np.float32)
    q=np.clip(np.rint((x*inverse[...,None]).astype(np.float32)),-7,7).astype(np.int8)
    return pack(q),scale

def reconstruct(packed,scale):
    return (unpack(packed).astype(np.float32)*np.asarray(scale,dtype=np.float32)[...,None]).astype(np.float32)
