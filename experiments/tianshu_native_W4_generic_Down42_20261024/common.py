# SPDX-License-Identifier: Apache-2.0
"""Current native Down W4 pack, real activation fixtures; no GPU libraries imported."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_native_dotsaddz_down42_20261023'
spec=importlib.util.spec_from_file_location('_W4_FP16_existing_quant',PARENT/'common.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
ROOT=Path('/dev/shm/tianshu_native_W4_generic_Down42_20261024')
CPU=ROOT/'cpu'
GPU=ROOT/"gpu"
sha,ahash,f32=parent.sha,parent.ahash,parent.f32
DEADLINE='2026-10-24T15:58:00+00:00'

def metrics(actual,reference,kind):
    actual=np.asarray(actual,np.float64);reference=np.asarray(reference,np.float64)
    error=actual-reference
    rms=float(np.sqrt(np.mean(reference*reference)))
    relative=float(np.linalg.norm(error)/max(np.linalg.norm(reference),1.e-30))
    rtol,atol,limit=(.012,.006,.006) if kind=='BF16' else (.003,.001,.001)
    bound=rtol*abs(reference)+atol*rms
    return dict(passed=bool(np.isfinite(actual).all() and np.isfinite(reference).all() and relative<=limit and np.all(abs(error)<=bound)),
        cells=int(actual.size),failed_cells=int(np.count_nonzero(abs(error)>bound)),relative_L2=relative,
        max_abs=float(abs(error).max()),reference_RMS=rms,rtol=rtol,atol_RMS=atol,relative_L2_limit=limit)
LAYERS=[0,20,41]
PHASES=[0,1,2]
VARIANTS=["BF16_dotsaddz_current","BF16_generic_prepacked"]
RELEASE_PAYLOADS=BASE.parent/'tianshu_W4_QKVO64_fitted_release_20261024/W4_PARENT_PAYLOADS.sha256'
FORMAT=BASE.parent/'tianshu_native_W4_formats_20261023'

def generic_arrays(w,s,z):
    q=parent.unpack(w)
    codes=q.T
    raw=((codes[:,::2]<<4)|codes[:,1::2]).astype(np.uint8)
    roundtrip=np.empty_like(codes)
    roundtrip[:,::2]=raw>>4;roundtrip[:,1::2]=raw&15
    assert np.array_equal(roundtrip,codes)
    offset=parent.bf16(f32(z)+8*f32(s))
    assert bool((offset==0).all()), 'This experiment only current symmetric Down'
    params=np.stack((s,offset),axis=-1)
    old=parent.bf16(q.astype(np.float32)*np.repeat(f32(s),128,axis=0)+np.repeat(f32(z),128,axis=0))
    generic=parent.bf16((q.astype(np.float32)-8)*np.repeat(f32(params[...,0]),128,axis=0)+np.repeat(f32(params[...,1]),128,axis=0))
    assert np.array_equal(old,generic)
    return raw,params,dict(logical_q_sha256=ahash(q),raw_high_even_sha256=ahash(raw),
        params_sha256=ahash(params),dequantized_BF16_sha256=ahash(old),
        same_quantized_weight_words=True,full_high_even_roundtrip_exact=True,
        raw_shape=list(raw.shape),params_shape=list(params.shape),expected_native_pack_shape=[256,48,32,4])
