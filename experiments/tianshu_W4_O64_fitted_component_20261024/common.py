# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json,struct
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/dev/shm/tianshu_W4_O64_fitted_component_20261024');CPU=ROOT/'cpu';GPU=ROOT/'gpu';MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');ROLES=['O'];MODES=['minmax','fitted'];VARIANTS=['BF16_actual_parent','W4_minmax','W4_fitted'];LAYERS=[0,20,41];PHASES=[0,1,2];DEADLINE='2026-10-24T15:58:00+00:00'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bf16(x):
 b=np.asarray(x,np.float32).view(np.uint32);return ((b+np.uint32(32767)+((b>>16)&1))>>16).astype(np.uint16)
def f32(x):return (np.asarray(x,np.uint16).astype(np.uint32)<<16).view(np.float32)
def original(layer,role):
 with MODEL.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
 def tensor(part):
  item=h[f'model.layers.{layer}.self_attn.{part}_proj.weight'];assert item['dtype']=='BF16';return np.memmap(MODEL,mode='r',dtype='<u2',offset=8+n+item['data_offsets'][0],shape=tuple(item['shape']))
 a=np.concatenate([tensor(p) for p in ('q','k','v')],axis=0) if role=='QKV' else np.asarray(tensor('o'))
 assert a.shape==(2560 if role=='QKV' else 2048,2048);return np.ascontiguousarray(a)
def pack(q):
 v=q.reshape(q.shape[0],-1,8).astype(np.uint32);return np.sum(v<<(4*np.arange(8,dtype=np.uint32)),axis=-1,dtype=np.uint32).view(np.int32)
def unpack(p):return (((p.view(np.uint32)[...,None]>>(4*np.arange(8,dtype=np.uint32)))&15).reshape(p.shape[0],-1)).astype(np.uint8)
def quant(w,mode):
 n,k=w.shape;v=f32(w).reshape(n,k//128,128)
 if mode=='sym':
  s=bf16(np.maximum(abs(v).max(2),np.float32(1e-8))/np.float32(7));z=bf16(-8*f32(s));q=(np.clip(np.rint(v/f32(s)[:,:,None]),-8,7).astype(np.int8)+8).astype(np.uint8)
 else:
  lo=v.min(2);hi=v.max(2);z=bf16(lo);s=bf16(np.maximum(hi-lo,np.float32(1e-8))/np.float32(15));q=np.clip(np.rint((v-f32(z)[:,:,None])/f32(s)[:,:,None]),0,15).astype(np.uint8)
 q=q.reshape(n,k).T.copy();return q,s.T.copy(),z.T.copy(),pack(q)
def dequant(q,s,z):return bf16(q.astype(np.float32)*np.repeat(f32(s),128,axis=0)+np.repeat(f32(z),128,axis=0))
def inputs(layer,role,phase):
 seed=202610240000+layer*100+ROLES.index(role)*10+phase;a=np.random.default_rng(seed).standard_normal((64,2048),dtype=np.float32);a[0]=0;return bf16(a)
def compare(words,reference):
 a=f32(words).astype(np.float64);d=a-reference;rms=float(np.sqrt(np.mean(reference*reference)));relative=float(np.linalg.norm(d)/max(np.linalg.norm(reference),1e-30));bound=.012*abs(reference)+.006*rms;return dict(pass_numeric=bool(np.isfinite(a).all() and np.isfinite(reference).all() and relative<=.006 and np.all(abs(d)<=bound)),relative_L2=relative,failed_cells=int(np.count_nonzero(abs(d)>bound)),cells=int(a.size),max_abs=float(abs(d).max()))
