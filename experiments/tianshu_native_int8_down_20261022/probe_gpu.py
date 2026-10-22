# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args();remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>1800;signal.alarm(min(remain,1800));root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(root/'hardware_before.txt').write_text(hw)
 sources=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text())['sources'];assert all(sha(p)==h for p,h in sources.items())
 for p,h in sources.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,cases=[],rounds=[],model_executed=False,official_benchmark=False,adopted=False,new_precision_approximation=True,checkpoint_sha256_before=sha(MODEL))
 def save():
  temp=root/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'result.json')
 save()
 try:
  import torch,vllm_fl,triton
  from safetensors import safe_open
  torch.set_num_threads(4);quant=load('_down_quant',ROOT/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py');dense=load('_down_reference',ROOT/'tianshu_dense_w8a8_20261020/kernel.py');epi=load('_down_gateup_epi',ROOT/'tianshu_native_int8_prefill_v2_20261022/kernel.py');dequant=load('_down_dequant',ROOT/'tianshu_native_int8_lm64_20261022/kernel.py')
  prior=json.loads((ROOT/'tianshu_native_int8_prefill_v2_20261022/GPU_GATE_32g.json').read_text());anchors={c['M']:c['variants']['NN_R1']['compiled']['binary_sha256'] for c in prior['cases'] if c['M'] in (512,4096)};state['target']=str(triton.runtime.driver.active.get_current_target())
  with safe_open(str(MODEL),framework='pt',device='cpu') as checkpoint:gateup=torch.cat([checkpoint.get_tensor(f'model.layers.0.mlp.{r}_proj.weight') for r in ('gate','up')],0).contiguous();down=checkpoint.get_tensor('model.layers.0.mlp.down_proj.weight').contiguous()
  def pack(t):v=t.float();s=v.abs().amax(1).clamp_min(1.e-8)/127.;return (v/s[:,None]).round().clamp(-127,127).to(torch.int8),s
  def thash(t):return ahash(t.contiguous().view(torch.uint8).numpy())
  gu_code,gu_scale=pack(gateup);code,scale=pack(down);assert tuple(down.shape)==(2048,6144);state['GateUp_signature']={key:thash(t) for key,t in [('original_BF16_sha256',gateup),('packed_sha256',gu_code),('scales_sha256',gu_scale)]};expected=next(r for r in json.loads((ROOT/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text()) if r['prefix']=='model.layers.0.mlp.gate_up_proj');assert all(expected[k]==v for k,v in state['GateUp_signature'].items());state['Down_signature']={key:thash(t) for key,t in [('original_BF16_sha256',down),('packed_sha256',code),('scales_sha256',scale)]}
  gu_w,gu_s=gu_code.T.contiguous().cuda(),gu_scale.cuda();dummy=torch.empty(1,device='cuda',dtype=torch.int32);diag_config=dict(bm=32,bn=128,bk=128,warps=2,stages=1)
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def metadata(c):return dict(hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared)
  def run_case(m):
   buffers=[];tensors={}
   for name,shape,dtype,value in [('a',(m,2048),torch.bfloat16,17.),('guq',(m,2048),torch.int8,17),('gus',(m,),torch.float32,17.),('gui',(m,12288),torch.int32,-777),('h',(m,6144),torch.bfloat16,17.),('q',(m,6144),torch.int8,17),('asc',(m,),torch.float32,17.),('wd',(2048,6144),torch.bfloat16,17.),('wnt',(2048,6144),torch.int8,17),('wnn',(6144,2048),torch.int8,17),('ws',(2048,),torch.float32,17.),('integer',(m,2048),torch.int32,-777),('refint',(m,2048),torch.int32,-777),('diag',(m,2048),torch.bfloat16,17.),('control',(m,2048),torch.bfloat16,17.)]+[(f'{layout}_B{block}',(m,2048),torch.bfloat16,17.) for layout in ('NT','NN') for block in (512,1024)]:
    count=int(np.prod(shape));store=torch.full((count+512,),value,dtype=dtype,device='cuda');tensors[name]=store[256:count+256].view(shape);buffers.append((store,value))
   a,h,q,asc,integer=tensors['a'],tensors['h'],tensors['q'],tensors['asc'],tensors['integer'];wd,wnt,wnn,ws=tensors['wd'],tensors['wnt'],tensors['wnn'],tensors['ws'];sample=np.random.default_rng(202610221500+m).standard_normal((m,2048),dtype=np.float32);bits=sample.view(np.uint32);base=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);base[0]=0;ac=torch.from_numpy(base).view(torch.bfloat16)
   def fixture(phase):
    sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.;a.copy_(ac if phase==0 else ac*(.5 if phase==1 else -.5));wd.copy_(down*sign*factor);wnt.copy_(code*sign);wnn.copy_((code*sign).T.contiguous());ws.copy_(scale*factor);quant.quantize(a,tensors['guq'],tensors['gus']);torch._int_mm(tensors['guq'],gu_w,out=tensors['gui']);compiled=epi._epilogue[(m,6)](tensors['gui'],tensors['gus'],gu_s,h,h,m,6144,1,False,num_warps=8,num_stages=1);torch.cuda.synchronize()
    if m in anchors:assert metadata(compiled)['binary_sha256']==anchors[m]
    return compiled
   fixture_kernel=fixture(0)
   def control():torch.mm(h,wd.T,out=tensors['control'])
   operations={'control':control};variants={}
   for layout in ('NT','NN'):
    for block in (512,1024):
     name=f'{layout}_B{block}';weight=wnt.T if layout=='NT' else wnn;out=tensors[name]
     def operation(weight=weight,out=out,block=block):quant.quantize(h,q,asc);torch._int_mm(q,weight,out=integer);return dequant._dequant[(triton.cdiv(m*2048,block),)](integer,asc,ws,out,m,2048,block,num_warps=4,num_stages=1)
     compiled=operation();operations[name]=operation;variants[name]=dict(layout=layout,block=block,compiled=metadata(compiled))
   control();torch.cuda.synchronize();graphs={}
   for name,fn in operations.items():
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn()
    graphs[name]=graph
   case=dict(M=m,K=6144,N=2048,variants=variants,GateUp_fixture_epilogue=metadata(fixture_kernel),native_workspace_bytes=m*2048*4,phases=[],complete=False);state['cases'].append(case);state['phase']=f'M{m}';save();rows=np.unique(np.linspace(0,m-1,8).astype(np.int64));cols=np.unique(np.concatenate([np.linspace(0,2047,32).astype(np.int64),np.arange(2040,2048)]));rg=torch.tensor(rows,device='cuda');cg=torch.tensor(cols,device='cuda')
   for phase in range(3):
    fixture(phase);cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();immutable=[x.clone() for x in (a,h,wd,wnt,wnn,ws)];quant.quantize(h,q,asc);dense.mm(q,wnt,asc,ws,tensors['diag'],tensors['refint'],diag_config,True);torch.cuda.synchronize();quant_words=words(tensors['diag']);reference_hash=ahash(quant_words);entries=[]
    for name,fn in operations.items():
     fn();torch.cuda.synchronize();eager=words(tensors[name]);tensors[name].fill_(float('nan'));integer.fill_(-2147483647);graphs[name].replay();torch.cuda.synchronize();assert np.array_equal(words(tensors[name]),eager) and bool(torch.isfinite(tensors[name]).all())
     if name!='control':assert torch.equal(integer,tensors['refint']) and np.array_equal(eager,quant_words)
     entries.append(dict(name=name,full_INT32_exact=(name!='control'),full_quantized_BF16_exact=(name!='control'),output_sha256=ahash(eager),graph_poison_eager_equal=True))
    error=tensors['NT_B512'].float()-tensors['control'].float();relative_l2=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(tensors['control'].float()).clamp_min(1e-20));maxerror=float(error.abs().max());assert np.isfinite(relative_l2) and np.isfinite(maxerror)
    for current,before in zip((a,h,wd,wnt,wnn,ws),immutable):assert torch.equal(current,before)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    for store,value in buffers:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
    path=root/f'M{m}_phase{phase}.npz';np.savez(path,H_words=words(h),rows=rows,columns=cols,integer_selected=tensors['refint'][rg[:,None],cg[None,:]].cpu().numpy(),candidate_selected_words=quant_words[np.ix_(rows,cols)],control_selected_words=words(tensors['control'][rg[:,None],cg[None,:]]))
    case['phases'].append(dict(phase=phase,weight_sign=1 if phase==0 else -1,weight_scale_factor=.5 if phase==2 else 1.,artifact=path.name,sha256=sha(path),H_words_sha256=ahash(words(h)),Aq_sha256=thash(q.cpu()),As_sha256=thash(asc.cpu()),WS_sha256=thash(ws.cpu()),integer_sha256=thash(tensors['refint'].cpu()),quantized_output_sha256=reference_hash,relative_L2_error_vs_original_BF16=relative_l2,maximum_absolute_error_vs_original_BF16=maxerror,entries=entries,guards_inputs_weights_RNG=True));save()
   for mode in ('graph','eager'):
    for roundid in range(7):
     names=list(operations);shift=roundid%len(names);names=names[shift:]+names[:shift]
     if roundid%2:names.reverse()
     times={}
     for name in names:
      fn=graphs[name].replay if mode=='graph' else operations[name];begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
      for _ in range(10):fn()
      end.record();end.synchronize();times[name]=begin.elapsed_time(end)/10
     state['rounds'].append(dict(M=m,mode=mode,round=roundid,order=names,replays=10,milliseconds=times))
   for store,value in buffers:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
   case['complete']=True;save();print(json.dumps(dict(M=m,complete=True,relative_L2=[r['relative_L2_error_vs_original_BF16'] for r in case['phases']])),flush=True)
  for m in (64,128,512,4096):run_case(m);torch.cuda.empty_cache()
  assert thash(gu_w.T.cpu().contiguous())==state['GateUp_signature']['packed_sha256'] and thash(gu_s.cpu())==state['GateUp_signature']['scales_sha256'];state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_before']==state['checkpoint_sha256_after'] and all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='complete',checkpoint_unchanged=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
