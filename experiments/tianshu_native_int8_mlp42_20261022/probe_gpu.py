# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args();remain=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remain>1800;signal.alarm(min(remain,1800))
 root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(root/'hardware_before.txt').write_text(hw)
 sources=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text())['sources'];assert all(sha(path)==digest for path,digest in sources.items())
 for path,digest in sources.items():
  snapshot=root/'source_snapshot'/Path(path).relative_to('/');snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(path).read_bytes());assert sha(snapshot)==digest
 state=dict(complete=False,phase='initializing',runner_pid=os.getpid(),frozen_sources=sources,cases=[],rounds=[],banks=[],model_executed=False,official_benchmark=False,adopted=False,checkpoint_sha256_before=sha(MODEL))
 def save():
  tmp=root/'result.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'result.json')
 save()
 try:
  import torch,vllm_fl,flag_gems,triton
  from safetensors import safe_open
  torch.set_num_threads(4);quant=load('_native_mlp42_quant',ROOT/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py');old=load('_native_mlp42_current',ROOT/'tianshu_nn_silu_large_tiles_v2_20261021/kernel.py');new=load('_native_mlp42_candidate',ROOT/'tianshu_native_int8_prefill_v2_20261022/kernel.py')
  selected=json.loads((ROOT/'tianshu_m_range_prefill_service_v2_20261021/selected_plan.json').read_text());component=json.loads(Path(selected['component_gate']).read_text());positive=json.loads((ROOT/'tianshu_native_int8_prefill_v2_20261022/GPU_GATE_32g.json').read_text());signatures={r['prefix']:r for r in json.loads((ROOT/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text())}
  def fingerprint(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
  def ahash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
  def metadata(c):return dict(hash=c.hash,binary_sha256=hashlib.sha256(c.kernel).hexdigest(),n_regs=c.n_regs,n_spills=c.n_spills,shared=c.metadata.shared)
  banks=[]
  with safe_open(str(MODEL),framework='pt',device='cpu') as checkpoint:
   for index in range(42):
    prefix=f'model.layers.{index}.mlp.';w=torch.cat([checkpoint.get_tensor(prefix+role+'_proj.weight') for role in ('gate','up')],0).contiguous();v=w.float();scale=v.abs().amax(1).clamp_min(1.e-8)/127.;code=(v/scale[:,None]).round().clamp(-127,127).to(torch.int8);down=checkpoint.get_tensor(prefix+'down_proj.weight').contiguous();proof=dict(layer=index,prefix=prefix+'gate_up_proj',original_BF16_sha256=fingerprint(w),packed_sha256=fingerprint(code),scales_sha256=fingerprint(scale),Down_BF16_sha256=fingerprint(down));assert all(proof[key]==signatures[proof['prefix']][key] for key in ('original_BF16_sha256','packed_sha256','scales_sha256'));state['banks'].append(proof);banks.append(dict(W=code.T.contiguous().cuda(),S=scale.cuda(),D=down.cuda()));assert fingerprint(banks[-1]['W'].T.cpu().contiguous())==proof['packed_sha256'] and fingerprint(banks[-1]['S'].cpu())==proof['scales_sha256'] and fingerprint(banks[-1]['D'].cpu())==proof['Down_BF16_sha256'];print(json.dumps(dict(bank=index,loaded=True)),flush=True)
  state['GPU_bank_bytes']=sum(t.numel()*t.element_size() for bank in banks for t in bank.values());state['phase']='banks_loaded';save()
  def guarded(shape,dtype,value):
   count=int(np.prod(shape));storage=torch.full((count+512,),value,dtype=dtype,device='cuda');return storage,storage[256:count+256].view(shape),value
  def guards(buffers):
   for storage,_,value in buffers:assert bool((storage[:256]==value).all() and (storage[-256:]==value).all())
  for m in (512,4096):
   k,n,i=2048,12288,6144;config=selected['configuration_by_M'][str(m)];buffers=[];arrays={};sample=np.random.default_rng(202610221300+m).standard_normal((m,k),dtype=np.float32);u=sample.view(np.uint32);base_words=((u+32767+((u>>16)&1))>>16).astype(np.uint16);base_words[0]=0;ac=torch.from_numpy(base_words).view(torch.bfloat16)
   for field,shape,dtype,value in [('a',(m,k),torch.bfloat16,17.),('q',(m,k),torch.int8,17),('as',(m,),torch.float32,17.),('old_h',(m,i),torch.bfloat16,17.),('new_h',(m,i),torch.bfloat16,17.),('old_i',(m,n),torch.int32,-777),('new_i',(m,n),torch.int32,-777),('old_p',(m,n),torch.bfloat16,17.),('new_p',(m,n),torch.bfloat16,17.)]:
    buf=guarded(shape,dtype,value);buffers.append(buf);arrays[field]=buf[1]
   a,q,asc=arrays['a'],arrays['q'],arrays['as'];a.copy_(ac);dummy=torch.empty(1,device='cuda',dtype=torch.int32);outputs={}
   for mode in ('current','native'):
    outputs[mode]=[]
    for _ in range(42):buf=guarded((m,2048),torch.bfloat16,17.);buffers.append(buf);outputs[mode].append(buf[1])
   def current_layer(index):
    bank=banks[index];quant.quantize(a,q,asc);compiled=old.fused(q,bank['W'],asc,bank['S'],arrays['old_h'],dummy,arrays['old_h'],config);flag_gems.mm_out(arrays['old_h'],bank['D'].T,out=outputs['current'][index]);return compiled
   def native_layer(index):
    bank=banks[index];quant.quantize(a,q,asc);torch._int_mm(q,bank['W'],out=arrays['new_i']);compiled=new.epilogue(arrays['new_i'],asc,bank['S'],arrays['new_h'],arrays['new_h'],1);flag_gems.mm_out(arrays['new_h'],bank['D'].T,out=outputs['native'][index]);return compiled
   current_binary=current_layer(0);native_binary=native_layer(0);torch.cuda.synchronize();assert metadata(current_binary)['binary_sha256']==component['actual_timed_candidate_kernels'][str(m)]['binary_sha256'];expected_new=next(c for c in positive['cases'] if c['M']==m)['variants']['NN_R1']['compiled']['binary_sha256'];assert metadata(native_binary)['binary_sha256']==expected_new
   for mode,operation in [('current',current_layer),('native',native_layer)]:
    for index in range(42):operation(index)
   torch.cuda.synchronize();graphs={}
   for mode,operation in [('current',current_layer),('native',native_layer)]:
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     for index in range(42):operation(index)
    graphs[mode]=graph
   case=dict(M=m,configuration=config,current_compiled=metadata(current_binary),native_epilogue_compiled=metadata(native_binary),phases=[],weight_banks_per_graph=42);state['cases'].append(case);state['phase']=f'M{m}';save();rows=np.unique(np.linspace(0,m-1,4).astype(np.int64));columns=np.unique(np.linspace(0,6143,32).astype(np.int64));both=np.concatenate([columns,columns+6144]);down_columns=np.unique(np.linspace(0,2047,16).astype(np.int64));rd,cd,bc,dc=[torch.from_numpy(v).cuda() for v in (rows,columns,both,down_columns)]
   for phase in range(3):
    actual=ac if phase==0 else ac*(.5 if phase==1 else -.5);a.copy_(actual);before=a.clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();eager_hashes=[];selected_artifacts=[]
    for index,bank in enumerate(banks):
     current_layer(index);native_layer(index);torch.cuda.synchronize();assert torch.equal(arrays['old_h'].view(torch.int16),arrays['new_h'].view(torch.int16)) and torch.equal(outputs['current'][index].view(torch.int16),outputs['native'][index].view(torch.int16)) and bool(torch.isfinite(outputs['native'][index]).all());timed_hidden=arrays['old_h'].clone();old.fused(q,bank['W'],asc,bank['S'],arrays['old_h'],arrays['old_i'],arrays['old_p'],config,True);new.epilogue(arrays['new_i'],asc,bank['S'],arrays['new_h'],arrays['new_p'],1,True);torch.cuda.synchronize();assert torch.equal(arrays['old_i'],arrays['new_i']) and torch.equal(arrays['old_p'].view(torch.int16),arrays['new_p'].view(torch.int16)) and torch.equal(arrays['old_h'].view(torch.int16),timed_hidden.view(torch.int16)) and torch.equal(arrays['new_h'].view(torch.int16),timed_hidden.view(torch.int16));eager_hashes.append(ahash(words(outputs['native'][index])))
     if index in (0,7,20,41):
      path=root/f'M{m}_phase{phase}_layer{index}.npz';np.savez(path,rows=rows,columns=columns,down_columns=down_columns,integer=arrays['new_i'][rd[:,None],bc[None,:]].cpu().numpy(),projected_words=words(arrays['new_p'][rd[:,None],bc[None,:]]),activation_selected_words=words(arrays['new_h'][rd[:,None],cd[None,:]]),Down_input_rows_words=words(arrays['new_h'][rd]),Down_output_selected_words=words(outputs['native'][index][rd[:,None],dc[None,:]]));selected_artifacts.append(dict(layer=index,artifact=path.name,sha256=sha(path)))
    fixture=root/f'M{m}_phase{phase}_fixture.npz';np.savez(fixture,A_words=words(a),Aq=q.cpu().numpy(),As=asc.cpu().numpy())
    for mode in outputs:
     for out in outputs[mode]:out.fill_(float('nan'))
    arrays['old_h'].fill_(float('nan'));arrays['new_h'].fill_(float('nan'));graphs['current'].replay();graphs['native'].replay();torch.cuda.synchronize()
    for index in range(42):assert bool(torch.isfinite(outputs['native'][index]).all()) and torch.equal(outputs['current'][index].view(torch.int16),outputs['native'][index].view(torch.int16)) and ahash(words(outputs['native'][index]))==eager_hashes[index]
    assert torch.equal(a,before) and torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng);guards(buffers);case['phases'].append(dict(phase=phase,fixture=fixture.name,fixture_sha256=sha(fixture),selected_bank_artifacts=selected_artifacts,all42_full_GateUp_INT32_projection_Down_exact=True,all42_whole_graph_poison_eager_hashes=eager_hashes,guards_input_RNG=True));save();print(json.dumps(dict(M=m,phase=phase,all42=True)),flush=True)
   for roundid in range(7):
    order=['current','native'] if roundid%2==0 else ['native','current'];times={}
    for mode in order:
     begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
     for _ in range(3):graphs[mode].replay()
     end.record();end.synchronize();times[mode]=begin.elapsed_time(end)/3/42
    state['rounds'].append(dict(M=m,round=roundid,order=order,replays=3,banks_per_graph=42,milliseconds_per_MLP_layer=times))
   guards(buffers);case['complete']=True;save();print(json.dumps(dict(M=m,complete=True)),flush=True)
   del buffers,arrays,buf,a,q,asc,outputs,out,graphs,graph,before,timed_hidden;torch.cuda.empty_cache()
  for index,bank in enumerate(banks):
   proof=state['banks'][index];assert fingerprint(bank['W'].T.cpu().contiguous())==proof['packed_sha256'] and fingerprint(bank['S'].cpu())==proof['scales_sha256'] and fingerprint(bank['D'].cpu())==proof['Down_BF16_sha256']
  state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_before']==state['checkpoint_sha256_after'] and all(sha(path)==digest for path,digest in sources.items());state.update(complete=True,phase='complete',checkpoint_unchanged=True,all42_bank_hashes_before_after=True);save()
 except BaseException as error:state.update(phase='failed',error=repr(error));save();raise
 finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
