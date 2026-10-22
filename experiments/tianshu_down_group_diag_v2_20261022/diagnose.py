# SPDX-License-Identifier: Apache-2.0
import ctypes,hashlib,importlib.util,json,os,signal,struct,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');FAILED=Path('/workspace/benchmark-results/20261022_down_group_GPU_tianshu');ROOT=Path('/workspace/benchmark-results/20261022_down_group_diag_v2_GPU_tianshu')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def main():
 remain=int((datetime.fromisoformat('2026-10-22T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert remain>1800
 def stop(signum,frame):raise TimeoutError(str(signum))
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(min(remain,1200));assert not ROOT.exists();ROOT.mkdir();hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(ROOT/'hardware_before.txt').write_text(hw)
 failed=read(FAILED/'result.json');assert failed['phase']=='failed' and failed['error']=='AssertionError()' and failed['cases'][0]['M']==64;sources=dict(failed['frozen_sources']);sources.update({str(p):sha(p) for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',FAILED/'result.json']});assert all(sha(p)==h for p,h in sources.items())
 for p,h in sources.items():
  target=ROOT/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',diagnostic_only=True,benchmark_allowed=False,adopted=False,original_failed_run_unchanged=True,frozen_sources=sources,original_failed_result_sha256=sha(FAILED/'result.json'),checkpoint_before=sha(MODEL),cases=[])
 def save():
  temp=ROOT/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(ROOT/'result.json')
 save()
 try:
  import torch,vllm_fl,triton
  from safetensors import safe_open
  torch.set_num_threads(4);original=load('_diag_exact_original_group_kernel',EXP/'tianshu_down_group_gpu_20261022/kernel.py');diag=load('_diag_v2_instrumented',BASE/'kernel.py');quant=load('_diag_original_input_quant',EXP/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py');fused=load('_diag_exact_GU64',EXP/'tianshu_nn_silu_two_dot_20261021/kernel.py')
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def f32(a):return (np.asarray(a,np.uint16).astype(np.uint32)<<16).view(np.float32)
  def bf16(a):
   bits=np.asarray(a,np.float32).view(np.uint32);return ((bits+32767+((bits>>16)&1))>>16).astype(np.uint16)
  def pack(t):v=t.float();s=v.abs().amax(1).clamp_min(1.e-8)/127.;return (v/s[:,None]).round().clamp(-127,127).to(torch.int8),s
  def thash(t):return ahash(t.cpu().contiguous().view(torch.uint8).numpy())
  def meta(c):return dict(hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared,compiler_metadata=str(c.metadata))
  def dump_asm(c,name):
   for key,value in c.asm.items():
    if key in ('llir','asm','ptx','amdgcn') and isinstance(value,str):(ROOT/f'{name}_{key}.txt').write_text(value)
  with safe_open(str(MODEL),framework='pt',device='cpu') as model:gu=torch.cat([model.get_tensor(f'model.layers.0.mlp.{r}_proj.weight') for r in ('gate','up')],0).contiguous();down=model.get_tensor('model.layers.0.mlp.down_proj.weight').contiguous()
  guq,gus=pack(gu);dq,ws=pack(down);assert thash(down)==failed['bank']['Down_original_BF16_sha256'] and thash(dq)==failed['bank']['Down_packed_sha256'] and thash(ws)==failed['bank']['Down_scales_sha256']
  with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
  info=header['model.layers.0.mlp.down_proj.weight'];weight_words=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(2048,6144));weight=f32(weight_words);scales=np.asarray(np.maximum(abs(weight).max(1),np.float32(1.e-8)).astype(np.float64)/127.,np.float32);packed=np.clip(np.rint(np.asarray(weight.astype(np.float64)/scales[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8);assert np.array_equal(packed,dq.numpy()) and np.array_equal(scales,ws.numpy())
  sample=np.random.default_rng(202610225800+64*10).standard_normal((64,2048),dtype=np.float32);sample=bf16(sample);sample[0]=0;a=torch.from_numpy(sample).view(torch.bfloat16).cuda();aq=torch.empty((64,2048),dtype=torch.int8,device='cuda');asc=torch.empty((64,),dtype=torch.float32,device='cuda');h=torch.empty((64,6144),dtype=torch.bfloat16,device='cuda');dummy=torch.empty(1,device='cuda',dtype=torch.int32);configuration=next(r for r in read(EXP/'tianshu_nn_silu_two_dot_20261021/GPU_GATE_32g.json')['best_by_shape'] if r['M']==64 and not r['strided']);quant.quantize(a,aq,asc);guw,gscale=guq.T.contiguous().cuda(),gus.cuda();compiled=fused.fused(aq,guw,asc,gscale,h,dummy,dummy.view(torch.bfloat16),configuration['configuration']);assert meta(compiled)['binary_sha256']==failed['cases'][0]['GateUp_binary_sha256'];torch.cuda.synchronize();hc=f32(words(h));wq,wscale=dq.cuda(),ws.cuda();hq=torch.empty_like(h,dtype=torch.int8);out=torch.empty((64,2048),device='cuda',dtype=torch.bfloat16);dbgout=torch.empty_like(out);acc=torch.empty_like(out,dtype=torch.float32);scaled=torch.empty_like(acc);libm=ctypes.CDLL('libm.so.6');fmaf=libm.fmaf;fmaf.argtypes=[ctypes.c_float]*3;fmaf.restype=ctypes.c_float
  for group in (128,512):
   count=6144//group;gs=torch.empty((64,count),device='cuda',dtype=torch.float32);original.quantize(h,hq,gs,group);torch.cuda.synchronize();qcpu=hq.cpu().numpy();scpu=gs.cpu().numpy();native=[];refsum=torch.zeros_like(acc);temporary=torch.empty((64,2048),device='cuda',dtype=torch.int32)
   for index,start in enumerate(range(0,6144,group)):
    torch._int_mm(hq[:,start:start+group].contiguous(),wq[:,start:start+group].contiguous().T,out=temporary);native.append(temporary.cpu().numpy().copy());refsum.add_(temporary.float()*gs[:,index,None])
   native=np.stack(native);reference=(refsum*wscale[None,:]).to(torch.bfloat16);refwords=words(reference);cpu_sep=np.zeros((64,2048),np.float32);cpu_fma=np.zeros_like(cpu_sep);cpu_int=[]
   for index,start in enumerate(range(0,6144,group)):
    block=hc[:,start:start+group];s=np.asarray(np.maximum(abs(block).max(1),np.float32(1.e-8)).astype(np.float64)/127.,np.float32);q=np.clip(np.rint(np.asarray(block.astype(np.float64)/s[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8);assert np.array_equal(s,scpu[:,index]) and np.array_equal(q,qcpu[:,start:start+group]);integer=q.astype(np.float64)@packed[:,start:start+group].astype(np.float64).T;assert np.array_equal(integer,np.rint(integer)) and np.max(abs(integer))<2**24;cpu_int.append(integer.astype(np.int32));cpu_sep=np.asarray(cpu_sep+integer.astype(np.float32)*s[:,None],np.float32);cpu_fma=np.asarray(cpu_fma.astype(np.float64)+integer*s[:,None].astype(np.float64),np.float32)
   cpu_int=np.stack(cpu_int);native_cpu_equal=np.array_equal(native,cpu_int);selected_rows=np.linspace(0,63,8).astype(np.int64);selected_cols=np.linspace(0,2047,32).astype(np.int64)
   for index,start in enumerate(range(0,6144,group)):
    integer=qcpu[selected_rows,start:start+group].astype(np.int64)@packed[selected_cols,start:start+group].astype(np.int64).T;assert np.array_equal(integer,cpu_int[index][np.ix_(selected_rows,selected_cols)])
   sepwords=bf16(np.asarray(cpu_sep*scales[None,:],np.float32));fmawords=bf16(np.asarray(cpu_fma*scales[None,:],np.float32));base_path=ROOT/f'G{group}_base.npz';np.savez(base_path,H_words=words(h),Aq=qcpu,AS=scpu,native_integer_groups=native,CPU_integer_groups=cpu_int,reference_words=refwords,reference_acc=refsum.cpu().numpy(),CPU_separate_acc=cpu_sep,CPU_fma_acc=cpu_fma,CPU_separate_words=sepwords,CPU_fma_words=fmawords)
   for warps in (2,4):
    c=original.mm(hq,wq,gs,wscale,out,group,warps);torch.cuda.synchronize();oldwords=words(out);assert meta(c)['binary_sha256']==failed['cases'][0]['compiled'][f'G{group}_w{warps}']['binary_sha256'];dump_asm(c,f'G{group}_w{warps}_original');record=dict(group=group,warps=warps,base_artifact=base_path.name,base_sha256=sha(base_path),original_compiled=meta(c),original_vs_native_BF16_mismatches=int(np.count_nonzero(oldwords!=refwords)),original_vs_CPU_separate_BF16_mismatches=int(np.count_nonzero(oldwords!=sepwords)),original_vs_CPU_fma_BF16_mismatches=int(np.count_nonzero(oldwords!=fmawords)),native_INT32_full_CPU_exact=native_cpu_equal,native_reference_vs_CPU_separate_BF16_mismatches=int(np.count_nonzero(refwords!=sepwords)),instrumented=[])
    for rn in (False,True):
     integers=torch.empty((count,64,2048),device='cuda',dtype=torch.int32);dc=diag.run(hq,wq,gs,wscale,dbgout,integers,acc,scaled,group,warps,rn);torch.cuda.synchronize();dw=words(dbgout);di=integers.cpu().numpy();path=ROOT/f'G{group}_w{warps}_RN{int(rn)}.npz';payload=dict(original_words=oldwords,debug_words=dw,debug_acc=acc.cpu().numpy(),debug_scaled=scaled.cpu().numpy())
     if not np.array_equal(di,native):payload['debug_integer_groups']=di
     np.savez(path,**payload);dump_asm(dc,f'G{group}_w{warps}_RN{int(rn)}');record['instrumented'].append(dict(explicit_RN=rn,compiled=meta(dc),integer_sha256=ahash(di),native_integer_sha256=ahash(native),debug_INT32_full_native_exact=np.array_equal(di,native),debug_INT32_full_CPU_exact=np.array_equal(di,cpu_int),BF16_mismatches_vs_original=int(np.count_nonzero(dw!=oldwords)),BF16_mismatches_vs_native_reference=int(np.count_nonzero(dw!=refwords)),BF16_mismatches_vs_CPU_separate=int(np.count_nonzero(dw!=sepwords)),BF16_mismatches_vs_CPU_fma=int(np.count_nonzero(dw!=fmawords)),FP32_acc_mismatches_vs_CPU_separate=int(np.count_nonzero(payload['debug_acc']!=cpu_sep)),FP32_acc_mismatches_vs_CPU_fma=int(np.count_nonzero(payload['debug_acc']!=cpu_fma)),artifact=path.name,artifact_sha256=sha(path)))
    state['cases'].append(record);save();print(json.dumps(dict(group=group,warps=warps,old_mismatch_native=record['original_vs_native_BF16_mismatches'],old_mismatch_CPU_sep=record['original_vs_CPU_separate_BF16_mismatches'],old_mismatch_CPU_fma=record['original_vs_CPU_fma_BF16_mismatches'],native_INT32_CPU_exact=native_cpu_equal,debug=[{k:d[k] for k in ('explicit_RN','debug_INT32_full_CPU_exact','BF16_mismatches_vs_CPU_separate','FP32_acc_mismatches_vs_CPU_fma','FP32_acc_mismatches_vs_CPU_separate')} for d in record['instrumented']])),flush=True)
  state['checkpoint_after']=sha(MODEL);assert state['checkpoint_before']==state['checkpoint_after'] and all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='diagnostic_complete',checkpoint_unchanged=True,candidate_not_qualified=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
