# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args();remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>2400
 def stopped(signum,frame):raise TimeoutError(f'signal {signum}')
 signal.signal(signal.SIGALRM,stopped);signal.signal(signal.SIGTERM,stopped);signal.alarm(min(remain,2400));root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
 sources=dict(read(BASE/'CPU_PREPARE_GATE_32g.json')['sources']);sources[str(BASE/'CPU_PREPARE_GATE_32g.json')]=sha(BASE/'CPU_PREPARE_GATE_32g.json');assert all(sha(p)==h for p,h in sources.items())
 for p,h in sources.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,cases=[],rounds=[],model_executed=False,official_benchmark=False,adopted=False,single_bank_cache_screen_only=True,layer=0,checkpoint_sha256_before=sha(MODEL),independent_fixture_seed=202610227600)
 def save():
  temp=root/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'result.json')
 save()
 try:
  import torch,vllm_fl,triton
  from safetensors import safe_open
  torch.set_num_threads(4);grouped=load('_group_down_fixed',BASE/'kernel.py');quant=load('_group_down_original_quant',EXP/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py');native=load('_group_down_original_GU',EXP/'tianshu_native_int8_prefill_v2_20261022/kernel.py');fused64=load('_group_down_original_GU64',EXP/'tianshu_nn_silu_two_dot_20261021/kernel.py');dequant=load('_group_down_original_dequant',EXP/'tianshu_native_int8_lm64_20261022/kernel.py')
  anchors={c['M']:c['variants']['NN_R1']['compiled']['binary_sha256'] for c in read(EXP/'tianshu_native_int8_prefill_v2_20261022/GPU_GATE_32g.json')['cases'] if c['M'] in (512,4096)};selected64=next(r for r in read(EXP/'tianshu_nn_silu_two_dot_20261021/GPU_GATE_32g.json')['best_by_shape'] if r['M']==64 and not r['strided'])
  def thash(t):return ahash(t.detach().cpu().contiguous().view(torch.uint8).numpy())
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def metadata(c):return dict(hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared)
  def pack(t):v=t.float();s=v.abs().amax(1).clamp_min(1.e-8)/127.;return (v/s[:,None]).round().clamp(-127,127).to(torch.int8),s
  with safe_open(str(MODEL),framework='pt',device='cpu') as checkpoint:gu=torch.cat([checkpoint.get_tensor(f'model.layers.0.mlp.{role}_proj.weight') for role in ('gate','up')],0).contiguous();down=checkpoint.get_tensor('model.layers.0.mlp.down_proj.weight').contiguous()
  guq,gus=pack(gu);dq,ds=pack(down);proof=dict(GU_original_BF16_sha256=thash(gu),GU_packed_sha256=thash(guq),GU_scales_sha256=thash(gus),Down_original_BF16_sha256=thash(down),Down_packed_sha256=thash(dq),Down_scales_sha256=thash(ds));old=read(Path(read(EXP/'tianshu_native_int8_down42_20261022/INDEPENDENT_CPU_GATE_32g.json')['raw_directory'])/'result.json')['banks'][0];assert all(proof[k]==old[k] for k in proof);state['bank']=proof;state['target']=str(triton.runtime.driver.active.get_current_target())
  guw,guscale=guq.T.contiguous().cuda(),gus.cuda();wd,wnt,wnn,ws=down.cuda(),dq.cuda(),dq.T.contiguous().cuda(),ds.cuda();weight_chunks={g:[wnt[:,i:i+g].contiguous().T for i in range(0,6144,g)] for g in (128,512)};dummy=torch.empty(1,device='cuda',dtype=torch.int32)
  def run_case(m):
   guards=[]
   def allocate(shape,dtype,value):
    count=int(np.prod(shape));store=torch.full((count+512,),value,dtype=dtype,device='cuda');guards.append((store,value));return store[256:count+256].view(shape)
   a=allocate((m,2048),torch.bfloat16,17.);aq=allocate((m,2048),torch.int8,17);asc=allocate((m,),torch.float32,17.);gui=allocate((m,12288),torch.int32,-777);h=allocate((m,6144),torch.bfloat16,17.);hq=allocate((m,6144),torch.int8,17);hs=allocate((m,),torch.float32,17.);integer=allocate((m,2048),torch.int32,-777);reference_sum=allocate((m,2048),torch.float32,17.);group_scales={g:allocate((m,6144//g),torch.float32,17.) for g in (128,512)}
   names=['control','ordinary_INT8']+[f'G{g}_w{w}' for g in (128,512) for w in (2,4)];outputs={name:allocate((m,2048),torch.bfloat16,17.) for name in names};block=512 if m==4096 else 1024
   def fixture(phase):
    sample=np.random.default_rng(202610227600+m*10+phase).standard_normal((m,2048),dtype=np.float32);bits=sample.view(np.uint32);bits=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);bits[0]=0;a.copy_(torch.from_numpy(bits).view(torch.bfloat16))
   def gateup():
    quant.quantize(a,aq,asc)
    if m==64:return fused64.fused(aq,guw,asc,guscale,h,dummy,dummy.view(torch.bfloat16),selected64['configuration'])
    torch._int_mm(aq,guw,out=gui);return native.epilogue(gui,asc,guscale,h,h,1)
   def control():torch.mm(h,wd.T,out=outputs['control'])
   def ordinary():
    quant.quantize(h,hq,hs);torch._int_mm(hq,wnn,out=integer);return dequant._dequant[(triton.cdiv(m*2048,block),)](integer,hs,ws,outputs['ordinary_INT8'],m,2048,block,num_warps=4,num_stages=1)
   operations=dict(control=control,ordinary_INT8=ordinary);fixture(0);expected=selected64['actual_timed_compiled']['binary_sha256'] if m==64 else anchors[m];assert metadata(gateup())['binary_sha256']==expected;compiled={}
   for group in (128,512):
    for warps in (2,4):
     name=f'G{group}_w{warps}';out=outputs[name]
     def operation(group=group,warps=warps,out=out):grouped.quantize(h,hq,group_scales[group],group);return grouped.mm(hq,wnt,group_scales[group],ws,out,group,warps)
     operations[name]=operation;compiled[name]=metadata(operation())
   for fn in operations.values():fn()
   torch.cuda.synchronize();graphs={}
   for scope in ('whole_MLP','Down_only'):
    for name,fn in operations.items():
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):
      if scope=='whole_MLP':gateup()
      fn()
     graphs[(scope,name)]=graph
   rows=np.unique(np.linspace(0,m-1,8).astype(np.int64));cols=np.unique(np.linspace(0,2047,32).astype(np.int64));rg=torch.tensor(rows,device='cuda');cg=torch.tensor(cols,device='cuda');case=dict(M=m,GateUp_binary_sha256=expected,compiled=compiled,phases=[],complete=False);state['cases'].append(case);state['phase']=f'M{m}';save()
   for phase in range(2):
    fixture(phase);gateup();before=a.clone();hidden=h.clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();control();ordinary();artifact=dict(rows=rows,columns=cols,H_selected_rows_words=words(h[rg]),control_selected_words=words(outputs['control'][rg[:,None],cg[None,:]]),ordinary_selected_words=words(outputs['ordinary_INT8'][rg[:,None],cg[None,:]]),ordinary_integer_selected=integer[rg[:,None],cg[None,:]].cpu().numpy());errors={};hashes={}
    for group in (128,512):
     grouped.quantize(h,hq,group_scales[group],group);artifact[f'G{group}_Aq_rows']=hq[rg].cpu().numpy();artifact[f'G{group}_AS_rows']=group_scales[group][rg].cpu().numpy();reference_sum.zero_();integer_groups=[];native_groups=torch.empty((6144//group,m,2048),device='cuda',dtype=torch.int32)
     for index,weight in enumerate(weight_chunks[group]):
      torch._int_mm(hq[:,index*group:(index+1)*group].contiguous(),weight,out=native_groups[index]);integer_groups.append(native_groups[index][rg[:,None],cg[None,:]].cpu().numpy())
     ref=torch.empty_like(outputs['control']);grouped.reference_fma(native_groups,group_scales[group],ws,ref);artifact[f'G{group}_integer_groups_selected']=np.stack(integer_groups);artifact[f'G{group}_reference_selected_words']=words(ref[rg[:,None],cg[None,:]])
     for warps in (2,4):
      name=f'G{group}_w{warps}';operations[name]();torch.cuda.synchronize();assert torch.equal(outputs[name],ref);artifact[name+'_selected_words']=words(outputs[name][rg[:,None],cg[None,:]])
    for name,output in outputs.items():
     assert bool(torch.isfinite(output).all());hashes[name]=ahash(words(output))
     if name!='control':errors[name]=float(torch.linalg.vector_norm(output.float()-outputs['control'].float())/torch.linalg.vector_norm(outputs['control'].float()).clamp_min(1e-20))
     output.fill_(float('nan'));h.fill_(float('nan'));graphs[('whole_MLP',name)].replay();torch.cuda.synchronize();assert ahash(words(output))==hashes[name]
    assert torch.equal(a,before) and torch.equal(h,hidden) and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    for store,value in guards:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
    path=root/f'M{m}_phase{phase}.npz';np.savez(path,**artifact);case['phases'].append(dict(phase=phase,seed=202610227600+m*10+phase,artifact=path.name,sha256=sha(path),full_grouped_BF16_equals_native_group_INT32_FP32_reference=True,output_hashes=hashes,relative_L2_vs_original_BF16=errors,graph_poison_eager_exact=True,guards_inputs_weights_RNG=True));save()
   for scope in ('whole_MLP','Down_only'):
    for mode in ('graph','eager'):
     for roundid in range(7):
      order=list(operations);shift=roundid%len(order);order=order[shift:]+order[:shift]
      if roundid%2:order.reverse()
      times={}
      for name in order:
       fn=graphs[(scope,name)].replay if mode=='graph' else operations[name];begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
       for _ in range(10):
        if mode=='eager' and scope=='whole_MLP':gateup()
        fn()
       end.record();end.synchronize();times[name]=begin.elapsed_time(end)/10
      state['rounds'].append(dict(M=m,scope=scope,mode=mode,round=roundid,order=order,replays=10,milliseconds=times))
   for store,value in guards:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
   case['complete']=True;save();print(json.dumps(dict(M=m,complete=True)),flush=True)
  for m in (64,512,4096):run_case(m);torch.cuda.empty_cache()
  assert thash(guw.T.contiguous())==proof['GU_packed_sha256'] and thash(guscale)==proof['GU_scales_sha256'] and thash(wd)==proof['Down_original_BF16_sha256'] and thash(wnt)==proof['Down_packed_sha256'] and thash(wnn.T.contiguous())==proof['Down_packed_sha256'] and thash(ws)==proof['Down_scales_sha256']
  for group,chunks in weight_chunks.items():assert thash(torch.cat([chunk.T for chunk in chunks],1))==proof['Down_packed_sha256']
  state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_after']==state['checkpoint_sha256_before'] and all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='complete',checkpoint_unchanged=True,all_bank_hashes_before_after=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:signal.alarm(0);(root/'hardware_after_internal.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
