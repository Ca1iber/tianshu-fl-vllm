# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
RANKS=Path('/workspace/benchmark-results/20261022_down_outlier_partition_CPU_tianshu/channel_ranks.npz')
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
 remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>2400
 def stopped(signum,frame):raise TimeoutError(f'signal {signum}')
 signal.signal(signal.SIGALRM,stopped);signal.signal(signal.SIGTERM,stopped);signal.alarm(min(remain,2400))
 root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
 hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
 sources=read(BASE/'CPU_PREPARE_GATE_32g.json')['sources'];sources=dict(sources);sources[str(BASE/'CPU_PREPARE_GATE_32g.json')]=sha(BASE/'CPU_PREPARE_GATE_32g.json');assert all(sha(p)==h for p,h in sources.items())
 for p,h in sources.items():
  target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,banks=[],cases=[],rounds=[],model_executed=False,official_benchmark=False,adopted=False,new_precision_approximation=True,checkpoint_sha256_before=sha(MODEL),independent_fixture_seed=202610224700,fixed_BF16_side_channels=128)
 def save():
  temporary=root/'result.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(root/'result.json')
 save()
 try:
  import torch,vllm_fl,triton
  from safetensors import safe_open
  torch.set_num_threads(4)
  quant=load('_side128_original_quant',EXP/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py')
  dense=load('_side128_independent_NT',EXP/'tianshu_dense_w8a8_20261020/kernel.py')
  native=load('_side128_original_GU_native',EXP/'tianshu_native_int8_prefill_v2_20261022/kernel.py')
  fused64=load('_side128_original_GU_M64',EXP/'tianshu_nn_silu_two_dot_20261021/kernel.py')
  dequant=load('_side128_original_Down_dequant',EXP/'tianshu_native_int8_lm64_20261022/kernel.py')
  side=load('_side128_candidate_fixed',BASE/'kernel.py')
  anchors={c['M']:c['variants']['NN_R1']['compiled']['binary_sha256'] for c in read(EXP/'tianshu_native_int8_prefill_v2_20261022/GPU_GATE_32g.json')['cases'] if c['M'] in (512,4096)}
  selected64=next(r for r in read(EXP/'tianshu_nn_silu_two_dot_20261021/GPU_GATE_32g.json')['best_by_shape'] if r['M']==64 and not r['strided'])
  old_banks=read(Path(read(EXP/'tianshu_native_int8_down42_20261022/INDEPENDENT_CPU_GATE_32g.json')['raw_directory'])/'result.json')['banks']
  with np.load(RANKS,allow_pickle=False) as data:ranks=data['top128_indices'].copy()
  assert ranks.shape==(42,128)
  def thash(t):return ahash(t.detach().cpu().contiguous().view(torch.uint8).numpy())
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def metadata(c):return dict(hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared)
  def pack(t):
   value=t.float();scale=value.abs().amax(1).clamp_min(1.e-8)/127.;return (value/scale[:,None]).round().clamp(-127,127).to(torch.int8),scale
  banks=[]
  with safe_open(str(MODEL),framework='pt',device='cpu') as checkpoint:
   for layer in range(42):
    gu=torch.cat([checkpoint.get_tensor(f'model.layers.{layer}.mlp.{role}_proj.weight') for role in ('gate','up')],0).contiguous();down=checkpoint.get_tensor(f'model.layers.{layer}.mlp.down_proj.weight').contiguous()
    guq,gus=pack(gu);dq,ds=pack(down);indices=torch.from_numpy(np.sort(ranks[layer]).astype(np.int32));assert indices.unique().numel()==128 and indices.min()>=0 and indices.max()<6144
    mask=torch.zeros(6144,dtype=torch.uint8);mask[indices.long()]=1;bulk=down.clone();bulk[:,indices.long()]=0;sq,ss=pack(bulk);sw=down[:,indices.long()].T.contiguous()
    proof=dict(layer=layer,GU_original_BF16_sha256=thash(gu),GU_packed_sha256=thash(guq),GU_scales_sha256=thash(gus),Down_original_BF16_sha256=thash(down),Down_packed_sha256=thash(dq),Down_scales_sha256=thash(ds),side_bulk_weight_sha256=thash(sq),side_bulk_scales_sha256=thash(ss),side_weight_sha256=thash(sw),side_indices_sha256=thash(indices),side_mask_sha256=thash(mask),sorted_indices=indices.tolist())
    assert all(proof[key]==old_banks[layer][key] for key in ('GU_original_BF16_sha256','GU_packed_sha256','GU_scales_sha256','Down_original_BF16_sha256','Down_packed_sha256','Down_scales_sha256'))
    bank=dict(GU=guq.T.contiguous().cuda(),GS=gus.cuda(),D=down.cuda(),DQ=dq.T.contiguous().cuda(),DS=ds.cuda(),SQ=sq.T.contiguous().cuda(),SS=ss.cuda(),SW=sw.cuda(),SI=indices.cuda(),SM=mask.cuda());banks.append(bank);state['banks'].append(proof)
    print(json.dumps(dict(layer=layer,loaded=True)),flush=True)
  state.update(phase='banks_loaded',target=str(triton.runtime.driver.active.get_current_target()),bank_payload_bytes=sum(t.numel()*t.element_size() for bank in banks for t in bank.values()));save()
  dummy=torch.empty(1,device='cuda',dtype=torch.int32);diag_config=dict(bm=32,bn=128,bk=128,warps=2,stages=1)

  def run_case(m):
   guards=[]
   def allocate(shape,dtype,value):
    count=int(np.prod(shape));store=torch.full((count+512,),value,dtype=dtype,device='cuda');guards.append((store,value));return store[256:count+256].view(shape)
   a=allocate((m,2048),torch.bfloat16,17.);aq=allocate((m,2048),torch.int8,17);asc=allocate((m,),torch.float32,17.);gui=allocate((m,12288),torch.int32,-777);h=allocate((m,6144),torch.bfloat16,17.);hq=allocate((m,6144),torch.int8,17);hs=allocate((m,),torch.float32,17.);integer=allocate((m,2048),torch.int32,-777);refint=allocate((m,2048),torch.int32,-777);diag=allocate((m,2048),torch.bfloat16,17.)
   side_ref=allocate((m,2048),torch.float32,17.);abs_sum=allocate((m,2048),torch.float32,17.);reference=allocate((m,2048),torch.float32,17.);bound=allocate((m,2048),torch.float32,17.)
   outputs={name:[allocate((m,2048),torch.bfloat16,17.) for _ in range(42)] for name in ('control','ordinary_INT8','side128')};block=512 if m==4096 else 1024
   def fixture(phase):
    sample=np.random.default_rng(202610224700+m*100+phase).standard_normal((m,2048),dtype=np.float32);bits=sample.view(np.uint32);bits=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);bits[0]=0;a.copy_(torch.from_numpy(bits).view(torch.bfloat16))
   def gateup(layer):
    bank=banks[layer];quant.quantize(a,aq,asc)
    if m==64:return fused64.fused(aq,bank['GU'],asc,bank['GS'],h,dummy,dummy.view(torch.bfloat16),selected64['configuration'])
    torch._int_mm(aq,bank['GU'],out=gui);return native.epilogue(gui,asc,bank['GS'],h,h,1)
   def control(layer):torch.mm(h,banks[layer]['D'].T,out=outputs['control'][layer])
   def ordinary(layer):
    bank=banks[layer];quant.quantize(h,hq,hs);torch._int_mm(hq,bank['DQ'],out=integer)
    return dequant._dequant[(triton.cdiv(m*2048,block),)](integer,hs,bank['DS'],outputs['ordinary_INT8'][layer],m,2048,block,num_warps=4,num_stages=1)
   def corrected(layer):
    bank=banks[layer];qkernel=side.quantize(h,bank['SM'],hq,hs);torch._int_mm(hq,bank['SQ'],out=integer);compiled=side.correct(integer,hs,bank['SS'],h,bank['SI'],bank['SW'],outputs['side128'][layer]);return qkernel,compiled
   operations=dict(control=control,ordinary_INT8=ordinary,side128=corrected);fixture(0)
   expected=selected64['actual_timed_compiled']['binary_sha256'] if m==64 else anchors[m]
   for layer in range(42):
    assert metadata(gateup(layer))['binary_sha256']==expected
    control(layer);ordinary(layer);qkernel,compiled=corrected(layer)
   torch.cuda.synchronize();graphs={}
   for scope in ('whole_MLP','Down_only_shared_last_H'):
    for name,operation in operations.items():
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):
      for layer in range(42):
       if scope=='whole_MLP':gateup(layer)
       operation(layer)
     graphs[(scope,name)]=graph
   case=dict(M=m,GateUp_binary_sha256=expected,quantize_compiled=metadata(qkernel),correct_compiled=metadata(compiled),phases=[],complete=False);state['cases'].append(case);state['phase']=f'M{m}';save()
   rows=np.unique(np.linspace(0,m-1,8).astype(np.int64));cols=np.unique(np.linspace(0,2047,32).astype(np.int64));rg=torch.tensor(rows,device='cuda');cg=torch.tensor(cols,device='cuda')
   for phase in range(2):
    fixture(phase);before=a.clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();hashes={name:[] for name in outputs};records=[]
    for layer,bank in enumerate(banks):
     gateup(layer);control(layer);ordinary(layer);corrected(layer);torch.cuda.synchronize();hidden=h.clone()
     dense.mm(hq,bank['SQ'].T.contiguous(),hs,bank['SS'],diag,refint,diag_config,True);torch.cuda.synchronize();assert torch.equal(integer,refint)
     side.reference(refint,hs,bank['SS'],h,bank['SI'],bank['SW'],side_ref,abs_sum,reference,bound);torch.cuda.synchronize()
     absolute_error=(outputs['side128'][layer].float()-reference).abs();assert bool((absolute_error<=bound).all()) and torch.equal(h,hidden)
     errors={}
     for name in outputs:
      output=outputs[name][layer];assert bool(torch.isfinite(output).all());hashes[name].append(ahash(words(output)))
      if name!='control':
       difference=output.float()-outputs['control'][layer].float();errors[name]=dict(squared_error=float(difference.square().sum()),squared_reference=float(outputs['control'][layer].float().square().sum()),relative_L2_error=float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(outputs['control'][layer].float()).clamp_min(1e-20)))
     path=root/f'M{m}_phase{phase}_layer{layer}.npz'
     np.savez(path,rows=rows,columns=cols,H_selected_rows_words=words(h[rg]),bulk_Aq_selected_rows=hq[rg].cpu().numpy(),bulk_As_selected_rows=hs[rg].cpu().numpy(),bulk_integer_selected=integer[rg[:,None],cg[None,:]].cpu().numpy(),control_selected_words=words(outputs['control'][layer][rg[:,None],cg[None,:]]),ordinary_selected_words=words(outputs['ordinary_INT8'][layer][rg[:,None],cg[None,:]]),candidate_selected_words=words(outputs['side128'][layer][rg[:,None],cg[None,:]]),reference_selected=reference[rg[:,None],cg[None,:]].cpu().numpy(),bound_selected=bound[rg[:,None],cg[None,:]].cpu().numpy())
     records.append(dict(layer=layer,artifact=path.name,sha256=sha(path),full_native_INT32_equals_independent_NT=True,full_corrected_output_inside_predeclared_error_bound=True,errors_vs_original_BF16=errors))
    for name in outputs:
     for output in outputs[name]:output.fill_(float('nan'))
     h.fill_(float('nan'));integer.fill_(-2147483647);graphs[('whole_MLP',name)].replay();torch.cuda.synchronize()
     for layer,output in enumerate(outputs[name]):assert bool(torch.isfinite(output).all()) and ahash(words(output))==hashes[name][layer]
    assert torch.equal(a,before) and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    for store,value in guards:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
    case['phases'].append(dict(phase=phase,seed=202610224700+m*100+phase,banks=records,eager_output_hashes=hashes,whole42_graph_poison_eager_verified=True,guards_inputs_RNG=True));save();print(json.dumps(dict(M=m,phase=phase,all42=True)),flush=True)
   for scope in ('whole_MLP','Down_only_shared_last_H'):
    for roundid in range(7):
     names=list(operations);shift=roundid%3;names=names[shift:]+names[:shift]
     if roundid%2:names.reverse()
     times={}
     for name in names:
      begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
      for _ in range(3):graphs[(scope,name)].replay()
      end.record();end.synchronize();times[name]=begin.elapsed_time(end)/3/42
     state['rounds'].append(dict(M=m,scope=scope,round=roundid,order=names,replays=3,banks_per_graph=42,milliseconds_per_bank=times))
   for store,value in guards:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
   case['complete']=True;save();print(json.dumps(dict(M=m,complete=True)),flush=True)
  for m in (64,512,4096):run_case(m);torch.cuda.empty_cache()
  for layer,bank in enumerate(banks):
   proof=state['banks'][layer]
   for key,tensor in [('GU_packed_sha256',bank['GU'].T.contiguous()),('GU_scales_sha256',bank['GS']),('Down_original_BF16_sha256',bank['D']),('Down_packed_sha256',bank['DQ'].T.contiguous()),('Down_scales_sha256',bank['DS']),('side_bulk_weight_sha256',bank['SQ'].T.contiguous()),('side_bulk_scales_sha256',bank['SS']),('side_weight_sha256',bank['SW']),('side_indices_sha256',bank['SI']),('side_mask_sha256',bank['SM'])]:assert thash(tensor)==proof[key]
  state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_before']==state['checkpoint_sha256_after'] and all(sha(p)==h for p,h in sources.items());state.update(complete=True,phase='complete',checkpoint_unchanged=True,all42_bank_hashes_before_after=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:signal.alarm(0);(root/'hardware_after_internal.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
