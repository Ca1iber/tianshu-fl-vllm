# SPDX-License-Identifier: Apache-2.0
"""Experimental complete norm/A8/native-QKV chain; not a model integration.

This module has no device work at import. The probe must instantiate Runtime only
once the other GPU trials are terminal and the card is idle. Guard allocations
are numeric-test-only; both timing arms allocate ordinary call-local outputs.
"""
import hashlib,importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
KINDS=('compiled_IR_A8_native_QKV','fused_IR_A8_native_QKV')

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def compiled_metadata(kernel):
 return dict(binary_sha256=hashlib.sha256(kernel.kernel).hexdigest(),cache_hash=kernel.hash,n_regs=kernel.n_regs,n_spills=kernel.n_spills,shared=kernel.metadata.shared)

class Runtime:
 def __init__(self):
  import torch
  from vllm.ir.ops import layernorm as ir
  self.torch=torch;self.preparation=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text());assert self.preparation['complete'] and self.preparation['three_QKV_weight_packs_independently_recomputed_exact']
  for path,digest in self.preparation['sources'].items():assert sha(path)==digest
  self.norm=load('_prefill_RMS_QKV_fixed_norm',EXP/'tianshu_IR_RMS_A8_20261023/kernel.py');self.dense=load('_prefill_RMS_QKV_original_A8',EXP/'tianshu_dense_w8a8_20261020/kernel.py');self.epilogue=load('_prefill_RMS_QKV_original_dequant',EXP/'tianshu_native_int8_lm64_20261022/kernel.py')
  native_norm=ir.rms_norm.impls['native'].impl_fn;native_add=ir.fused_add_rms_norm.impls['native'].impl_fn
  expected=Path('/usr/local/lib/python3.12/site-packages/vllm/ir/ops/layernorm.py');assert Path(native_norm.__code__.co_filename).resolve()==expected and Path(native_add.__code__.co_filename).resolve()==expected
  def original_norm(x,residual,weight):
   if residual is None:return native_norm(x,weight,1e-6,None),None
   return native_add(x,residual,weight,1e-6,None)
  self.compiled_norm=torch.compile(original_norm,backend='inductor',fullgraph=True,dynamic=False)
  self.compiled={};self.calls={};self.reference_scope='standalone compiled installed IR implementation; full-model norm CUBIN equivalence not established'

 def observe(self,key,kernel,expected_binary=None):
  if key in self.compiled:return
  metadata=compiled_metadata(kernel)
  if expected_binary is not None:assert metadata['binary_sha256']==expected_binary
  self.compiled[key]=metadata

 def __call__(self,kind,x,residual,norm_weight,packed,weight_scale,guarded=False):
  torch=self.torch;assert kind in KINDS;m,k=x.shape
  assert m in (512,4096) and k==2048 and x.is_cuda and x.dtype==torch.bfloat16 and x.is_contiguous()
  assert norm_weight.shape==(2048,) and norm_weight.device==x.device and norm_weight.dtype==torch.bfloat16 and norm_weight.is_contiguous()
  assert residual is None or (residual.shape==x.shape and residual.device==x.device and residual.dtype==torch.bfloat16 and residual.is_contiguous())
  assert packed.device==x.device and packed.shape==(2048,2560) and packed.dtype==torch.int8 and packed.is_contiguous()
  assert weight_scale.device==x.device and weight_scale.shape==(2560,) and weight_scale.dtype==torch.float32 and weight_scale.is_contiguous()
  guards=[]
  def allocate(shape,dtype):
   if not guarded:return torch.empty(shape,device=x.device,dtype=dtype)
   count=1
   for dimension in shape:count*=dimension
   tag=71 if dtype==torch.int8 else 31
   raw=torch.full((count+256,),tag,device=x.device,dtype=dtype);guards.append((raw,count,tag));return raw[128:128+count].view(shape)
  q=allocate((m,k),torch.int8);scale=allocate((m,),torch.float32)
  if kind=='compiled_IR_A8_native_QKV':
   normalized,residual_out=self.compiled_norm(x,residual,norm_weight);kernel=self.dense.quantize(normalized,q,scale);self.observe(f'baseline_A8_M{m}',kernel)
  else:
   normalized=allocate((m,k),torch.bfloat16);residual_out=allocate((m,k),torch.bfloat16) if residual is not None else None
   kernel=self.norm.run(x,residual,norm_weight,normalized,residual_out,q,scale);self.observe(f'fused_norm_M{m}_residual_{residual is not None}',kernel,self.preparation['expected_norm_binaries'][str(residual is not None)])
  # Identical original projection tail for both arms. Keep integer output alive
  # for full-byte graph/eager checks and independent INT64 sample references.
  integer=allocate((m,2560),torch.int32);projection=allocate((m,2560),torch.bfloat16)
  torch._int_mm(q,packed,out=integer)
  block=self.preparation['block_by_M'][str(m)];kernel=self.epilogue._dequant[((m*2560+block-1)//block,)](integer,scale,weight_scale,projection,m,2560,block,num_warps=4,num_stages=1)
  self.observe(f'QKV_dequant_M{m}',kernel,self.preparation['expected_QKV_dequant_binaries'][str(m)])
  key=(kind,m,residual is not None,bool(torch.cuda.is_current_stream_capturing()));self.calls[str(key)]=self.calls.get(str(key),0)+1
  return dict(norm=normalized,residual=residual_out,q=q,scale=scale,integer=integer,projection=projection),guards

 def check_guards(self,guards):
  torch=self.torch
  for raw,count,tag in guards:assert bool((raw[:128]==tag).all()) and bool((raw[128+count:]==tag).all())
