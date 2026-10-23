# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import hashlib,json,struct,numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;CPU=Path('/workspace/benchmark-results/20261023_native_W4_lm64_CPU_tianshu');GPU=Path('/workspace/benchmark-results/20261023_native_W4_lm64_GPU_tianshu');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');PROFILE=EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json';M,N,K=64,130560,2048;VARIANTS=['W8A8_current','W4_AWQ_RTN','W4_DOTS_RTN'];COLS=np.unique(np.linspace(0,N-1,512).astype(np.int64))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ahash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def bf16(x):
 b=np.asarray(x,np.float32).view(np.uint32);return ((b+32767+((b>>16)&1))>>16).astype(np.uint16)
def f32(x):return (np.asarray(x,np.uint16).astype(np.uint32)<<16).view(np.float32)
def weight():
 with MODEL.open('rb') as f:l=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(l))
 v=h['lm_head.weight'];assert v['shape']==[N,K] and v['dtype']=='BF16';return np.memmap(MODEL,mode='r',dtype='<u2',offset=8+l+v['data_offsets'][0],shape=(N,K))
def pack(q,order):
 q=q.reshape(q.shape[0],-1,8)[...,order].astype(np.uint32);return np.sum(q<<(4*np.arange(8,dtype=np.uint32)),axis=-1,dtype=np.uint32).view(np.int32)
def unpack(p,order):
 raw=(p.view(np.uint32)[...,None]>>(4*np.arange(8,dtype=np.uint32)))&15;r=np.empty_like(raw);r[...,order]=raw;return r.reshape(p.shape[0],-1).astype(np.uint8)
def quant_a(words):
 x=f32(words);s=np.asarray(np.maximum(abs(x).max(1),np.float32(1e-8)).astype(np.float64)/127.,np.float32);q=np.clip(np.rint(np.asarray(x.astype(np.float64)/s[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8);return q,s
def compare(actual,ref):
 a=f32(actual).astype(np.float64);d=a-ref;rms=float(np.sqrt(np.mean(ref*ref)));rl=float(np.linalg.norm(d)/max(np.linalg.norm(ref),1e-30));return dict(passed=bool(np.isfinite(a).all() and np.isfinite(ref).all() and np.all(abs(d)<=.012*abs(ref)+.006*rms) and rl<=.006),relative_L2=rl,max_abs=float(abs(d).max()),failed_cells=int(np.count_nonzero(abs(d)>.012*abs(ref)+.006*rms)),cells=int(a.size))
