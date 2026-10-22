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
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args();remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>1800;signal.alarm(min(remain,2400));root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(root/'hardware_before.txt').write_text(hw)
 sources=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text())['sources'];assert all(sha(p)==h for p,h in sources.items())
 for p,h in sources.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,banks=[],cases=[],rounds=[],model_executed=False,official_benchmark=False,adopted=False,new_precision_approximation=True,checkpoint_sha256_before=sha(MODEL))
 def save():
  temp=root/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'result.json')
 save()
 try:
  import torch,vllm_fl,triton
  from safetensors import safe_open
  torch.set_num_threads(4);quant=load('_d42_quant',ROOT/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py');dense=load('_d42_dense',ROOT/'tianshu_dense_w8a8_20261020/kernel.py');native=load('_d42_native',ROOT/'tianshu_native_int8_prefill_v2_20261022/kernel.py');fused64=load('_d42_fused64',ROOT/'tianshu_nn_silu_two_dot_20261021/kernel.py');fused128=load('_d42_fused128',ROOT/'tianshu_nn_silu_large_tiles_v2_20261021/kernel.py');dequant=load('_d42_dequant',ROOT/'tianshu_native_int8_lm64_20261022/kernel.py')
  anchor_gate=json.loads((ROOT/'tianshu_native_int8_prefill_v2_20261022/GPU_GATE_32g.json').read_text());anchors={c['M']:c['variants']['NN_R1']['compiled']['binary_sha256'] for c in anchor_gate['cases'] if c['M'] in (512,4096)};selection=json.loads((ROOT/'tianshu_m_range_prefill_service_v2_20261021/selected_plan.json').read_text());range_gate=json.loads(Path(selection['component_gate']).read_text());fused64gate=json.loads((ROOT/'tianshu_nn_silu_two_dot_20261021/GPU_GATE_32g.json').read_text());selected64=next(r for r in fused64gate['best_by_shape'] if r['M']==64 and not r['strided']);state['target']=str(triton.runtime.driver.active.get_current_target())
  def thash(t):return ahash(t.contiguous().view(torch.uint8).numpy())
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def metadata(c):return dict(hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared)
  def pack(t):v=t.float();s=v.abs().amax(1).clamp_min(1.e-8)/127.;return (v/s[:,None]).round().clamp(-127,127).to(torch.int8),s
  signatures={r['prefix']:r for r in json.loads((ROOT/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text())};banks=[]
  with safe_open(str(MODEL),framework='pt',device='cpu') as checkpoint:
   for layer in range(42):
    gu=torch.cat([checkpoint.get_tensor(f'model.layers.{layer}.mlp.{role}_proj.weight') for role in ('gate','up')],0).contiguous();down=checkpoint.get_tensor(f'model.layers.{layer}.mlp.down_proj.weight').contiguous();guq,gus=pack(gu);dq,ds=pack(down);prefix=f'model.layers.{layer}.mlp.gate_up_proj';proof=dict(layer=layer,prefix=prefix,GU_original_BF16_sha256=thash(gu),GU_packed_sha256=thash(guq),GU_scales_sha256=thash(gus),Down_original_BF16_sha256=thash(down),Down_packed_sha256=thash(dq),Down_scales_sha256=thash(ds));assert all(proof['GU_'+key]==signatures[prefix][key] for key in ('original_BF16_sha256','packed_sha256','scales_sha256'));state['banks'].append(proof);banks.append(dict(GU=guq.T.contiguous().cuda(),GS=gus.cuda(),D=down.cuda(),DQ=dq.T.contiguous().cuda(),DS=ds.cuda()));print(json.dumps(dict(layer=layer,loaded=True)),flush=True)
  dummy=torch.empty(1,device='cuda',dtype=torch.int32);diag_config=dict(bm=32,bn=128,bk=128,warps=2,stages=1);state.update(phase='banks_loaded',bank_payload_bytes=sum(t.numel()*t.element_size() for bank in banks for t in bank.values()));save()
  def run_case(m):
   buffers=[];tensors={}
   def allocate(name,shape,dtype,value):
    count=int(np.prod(shape));storage=torch.full((count+512,),value,dtype=dtype,device='cuda');tensor=storage[256:count+256].view(shape);buffers.append((storage,value));tensors[name]=tensor;return tensor
   a=allocate('a',(m,2048),torch.bfloat16,17.);aq=allocate('aq',(m,2048),torch.int8,17);ags=allocate('ags',(m,),torch.float32,17.);gui=allocate('gui',(m,12288),torch.int32,-777);h=allocate('h',(m,6144),torch.bfloat16,17.);hq=allocate('hq',(m,6144),torch.int8,17);hs=allocate('hs',(m,),torch.float32,17.);integer=allocate('integer',(m,2048),torch.int32,-777);refint=allocate('refint',(m,2048),torch.int32,-777);diag=allocate('diag',(m,2048),torch.bfloat16,17.)
   outputs={mode:[allocate(f'{mode}_{layer}',(m,2048),torch.bfloat16,17.) for layer in range(42)] for mode in ('control','candidate')};sample=np.random.default_rng(202610221600+m).standard_normal((m,2048),dtype=np.float32);bits=sample.view(np.uint32);base=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);base[0]=0;ac=torch.from_numpy(base).view(torch.bfloat16);a.copy_(ac);block=512 if m==4096 else 1024
   def gateup(layer):
    bank=banks[layer];quant.quantize(a,aq,ags)
    if m==64:return fused64.fused(aq,bank['GU'],ags,bank['GS'],h,dummy,dummy.view(torch.bfloat16),selected64['configuration'])
    if m==128:return fused128.fused(aq,bank['GU'],ags,bank['GS'],h,dummy,h,selection['configuration_by_M']['128'])
    torch._int_mm(aq,bank['GU'],out=gui);return native.epilogue(gui,ags,bank['GS'],h,h,1)
   def down_control(layer):torch.mm(h,banks[layer]['D'].T,out=outputs['control'][layer])
   def down_candidate(layer):
    bank=banks[layer];quant.quantize(h,hq,hs);torch._int_mm(hq,bank['DQ'],out=integer);return dequant._dequant[(triton.cdiv(m*2048,block),)](integer,hs,bank['DS'],outputs['candidate'][layer],m,2048,block,num_warps=4,num_stages=1)
   expected_binary=selected64['actual_timed_compiled']['binary_sha256'] if m==64 else range_gate['actual_timed_candidate_kernels']['128']['binary_sha256'] if m==128 else anchors[m]
   for layer in range(42):
    compiled=gateup(layer);assert metadata(compiled)['binary_sha256']==expected_binary;down_control(layer);new_compiled=down_candidate(layer)
   torch.cuda.synchronize();graphs={}
   for scope in ('whole_MLP','Down_only'):
    for mode,operation in [('control',down_control),('candidate',down_candidate)]:
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):
      for layer in range(42):
       if scope=='whole_MLP':gateup(layer)
       operation(layer)
     graphs[(scope,mode)]=graph
   case=dict(M=m,block=block,GateUp_binary_sha256=expected_binary,candidate_compiled=metadata(new_compiled),phases=[],complete=False);state['cases'].append(case);state['phase']=f'M{m}';save();rows=np.unique(np.linspace(0,m-1,8).astype(np.int64));cols=np.unique(np.linspace(0,2047,32).astype(np.int64));rg=torch.tensor(rows,device='cuda');cg=torch.tensor(cols,device='cuda')
   for phase in range(3):
    a.copy_(ac if phase==0 else ac*(.5 if phase==1 else -.5));before=a.clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();hashes={'control':[],'candidate':[]};records=[]
    for layer,bank in enumerate(banks):
     gateup(layer);down_control(layer);down_candidate(layer);torch.cuda.synchronize();hidden=h.clone();candidate_words=words(outputs['candidate'][layer]);quant.quantize(h,hq,hs);dense.mm(hq,bank['DQ'].T.contiguous(),hs,bank['DS'],diag,refint,diag_config,True);torch.cuda.synchronize();assert torch.equal(integer,refint) and np.array_equal(words(diag),candidate_words) and torch.equal(h,hidden)
     for mode in hashes:hashes[mode].append(ahash(words(outputs[mode][layer])))
     error=outputs['candidate'][layer].float()-outputs['control'][layer].float();relative=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(outputs['control'][layer].float()).clamp_min(1e-20));assert np.isfinite(relative) and bool(torch.isfinite(outputs['control'][layer]).all()) and bool(torch.isfinite(outputs['candidate'][layer]).all())
     path=root/f'M{m}_phase{phase}_layer{layer}.npz';np.savez(path,rows=rows,columns=cols,H_selected_rows_words=words(h[rg]),Aq_selected_rows=hq[rg].cpu().numpy(),As_selected_rows=hs[rg].cpu().numpy(),integer_selected=refint[rg[:,None],cg[None,:]].cpu().numpy(),candidate_selected_words=candidate_words[np.ix_(rows,cols)],control_selected_words=words(outputs['control'][layer][rg[:,None],cg[None,:]]));records.append(dict(layer=layer,artifact=path.name,sha256=sha(path),full_quantized_INT32_BF16_exact=True,relative_L2_error_vs_BF16=relative,max_absolute_error_vs_BF16=float(error.abs().max())))
    for mode in outputs:
     for output in outputs[mode]:output.fill_(float('nan'))
    h.fill_(float('nan'));graphs[('whole_MLP','control')].replay();h.fill_(float('nan'));integer.fill_(-2147483647);graphs[('whole_MLP','candidate')].replay();torch.cuda.synchronize()
    for mode in outputs:
     for layer,output in enumerate(outputs[mode]):assert bool(torch.isfinite(output).all()) and ahash(words(output))==hashes[mode][layer]
    assert torch.equal(a,before) and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    for store,value in buffers:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
    case['phases'].append(dict(phase=phase,banks=records,eager_output_hashes=hashes,whole42_graph_poison_eager_verified=True,guards_input_RNG=True));save();print(json.dumps(dict(M=m,phase=phase,all42=True)),flush=True)
   for scope in ('whole_MLP','Down_only'):
    for roundid in range(7):
     order=['control','candidate'] if roundid%2==0 else ['candidate','control'];times={}
     for mode in order:
      begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
      for _ in range(3):graphs[(scope,mode)].replay()
      end.record();end.synchronize();times[mode]=begin.elapsed_time(end)/3/42
     state['rounds'].append(dict(M=m,scope=scope,round=roundid,order=order,replays=3,banks_per_graph=42,milliseconds_per_bank=times))
   for store,value in buffers:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
   case['complete']=True;save();print(json.dumps(dict(M=m,complete=True)),flush=True)
  for m in (64,128,512,4096):run_case(m);torch.cuda.empty_cache()
  for layer,bank in enumerate(banks):
   proof=state['banks'][layer];assert thash(bank['GU'].T.cpu().contiguous())==proof['GU_packed_sha256'] and thash(bank['GS'].cpu())==proof['GU_scales_sha256'] and thash(bank['D'].cpu())==proof['Down_original_BF16_sha256'] and thash(bank['DQ'].T.cpu().contiguous())==proof['Down_packed_sha256'] and thash(bank['DS'].cpu())==proof['Down_scales_sha256']
  state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_before']==state['checkpoint_sha256_after'] and all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='complete',checkpoint_unchanged=True,all42_bank_hashes_before_after=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
