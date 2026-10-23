# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import hashlib,json,struct,numpy as np
BASE=Path(__file__).resolve().parent
CPU=Path('/workspace/benchmark-results/20261023_native_dotsaddz_down42_CPU_tianshu')
GPU=Path('/workspace/benchmark-results/20261023_native_dotsaddz_down42_GPU_tianshu')
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
PROFILE=BASE.parent/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json'
FIXTURE=Path('/workspace/benchmark-results/20261022_native_int8_down_GPU_tianshu')
ORDER=list(range(8));LAYERS=[0,20,41];PHASES=[0,1,2]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bf16(x):
 b=np.asarray(x,np.float32).view(np.uint32);return ((b+np.uint32(32767)+((b>>16)&1))>>16).astype(np.uint16)
def f32(x):return (np.asarray(x,np.uint16).astype(np.uint32)<<16).view(np.float32)
def header():
 with MODEL.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
 return n,h
def weight(layer):
 n,h=header();v=h[f'model.layers.{layer}.mlp.down_proj.weight'];assert v['dtype']=='BF16' and v['shape']==[2048,6144]
 return np.memmap(MODEL,mode='r',dtype='<u2',offset=8+n+v['data_offsets'][0],shape=(2048,6144))
def pack(q):
 v=q.reshape(q.shape[0],-1,8)[...,ORDER].astype(np.uint32);return np.sum(v<<(4*np.arange(8,dtype=np.uint32)),axis=-1,dtype=np.uint32).view(np.int32)
def unpack(p):
 v=(p.view(np.uint32)[...,None]>>(4*np.arange(8,dtype=np.uint32)))&15;r=np.empty_like(v);r[...,ORDER]=v;return r.reshape(p.shape[0],-1).astype(np.uint8)
def quant(w):
 wf=f32(w).reshape(2048,48,128);s=bf16(np.maximum(abs(wf).max(2),np.float32(1e-8))/np.float32(7));sf=f32(s);q=(np.clip(np.rint(wf/sf[:,:,None]),-8,7).astype(np.int8)+8).astype(np.uint8).reshape(2048,6144).T.copy();return q,s.T.copy(),pack(q),bf16(-8*f32(s.T))
def hwords(phase,layer):
 with np.load(FIXTURE/f'M64_phase{phase}.npz',allow_pickle=False) as f:return np.roll(f['H_words'],layer,axis=0).copy()
def compare(actual,ref):
 a=f32(actual).astype(np.float64);d=a-ref;rms=float(np.sqrt(np.mean(ref*ref)));rl=float(np.linalg.norm(d)/max(np.linalg.norm(ref),1e-30));bound=.012*abs(ref)+.006*rms;ok=bool(np.isfinite(a).all() and np.isfinite(ref).all() and np.all(abs(d)<=bound) and rl<=.006);return dict(pass_numeric=ok,relative_L2=rl,max_abs=float(abs(d).max()),reference_RMS=rms,failed_cells=int(np.count_nonzero(abs(d)>bound)),cells=int(a.size))
