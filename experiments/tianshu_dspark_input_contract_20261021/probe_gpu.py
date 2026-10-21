# SPDX-License-Identifier: Apache-2.0
"""Actual legacy DSpark host path + original fused GPU prepare, no model loaded."""
import argparse, json, os, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from common import BASE, OLD, CASES, oracle, sha

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);a=p.parse_args()
root=a.output;root.mkdir(exist_ok=False)
gate=json.loads((BASE/'CPU_GATE.json').read_text());assert gate['complete']
for path,h in gate['sources'].items():assert sha(path)==h
frozen=dict(gate['sources']);frozen[str(BASE/'CPU_GATE.json')]=sha(BASE/'CPU_GATE.json')
for path,h in frozen.items():
 target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==h
remaining=int((datetime.fromisoformat(a.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>0
signal.alarm(min(900,remaining))
assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
import torch
import vllm_fl
sys.path.insert(0,str(OLD))
from tianshu_dspark_adapter.proposer import DSparkProposer
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode import utils
assert Path(DSparkProposer.set_inputs_first_pass.__code__.co_filename).resolve()==OLD/'tianshu_dspark_adapter/proposer.py'
assert sha(DFlashProposer.set_inputs_first_pass.__code__.co_filename)==gate['sources'][str(Path(DFlashProposer.set_inputs_first_pass.__code__.co_filename))]
state=dict(complete=False,phase='running',pid=os.getpid(),model_executed=False,official_benchmark=False,frozen_sources=frozen,cases=[])
def save():
 temp=root/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'result.json')
save();generator=torch.Generator(device='cpu').manual_seed(202610210951)
def guarded(size,dtype):
 storage=torch.full((size+22,),-177,dtype=dtype,device='cuda');return storage[11:-11],storage
try:
 for case in CASES:
  for phase in (0,1):
   context=case['context'];past=[v+phase for v in case['past']];batch=len(context);seq=[c+s for c,s in zip(context,past)];total=sum(context)
   cols=(max(seq)+7+15)//16+3
   pagedata=(torch.randperm(batch*cols,generator=generator,dtype=torch.int64)+101).reshape(batch,cols).to(torch.int32)
   table_storage=torch.full((batch,cols+5),-177,dtype=torch.int32,device='cuda');table=table_storage[:,:cols];table.copy_(pagedata)
   bonus=torch.arange(batch,dtype=torch.int64)+1000+phase*100
   expected=oracle(context,past,case['rejected'],bonus.tolist(),pagedata.tolist())
   positions=torch.tensor(expected['context_positions'],dtype=torch.int64,device='cuda')
   ids=torch.arange(total,dtype=torch.int64,device='cuda')%130560
   hidden=torch.zeros((total,2048),dtype=torch.bfloat16,device='cuda')
   start=[0]
   for c in context:start.append(start[-1]+c)
   qstart=torch.tensor(start,dtype=torch.int32,device='cuda');seq_gpu=torch.tensor(seq,dtype=torch.int32,device='cuda')
   rejected=(torch.tensor(case['rejected'],dtype=torch.int32,device='cuda') if case['rejected'] is not None else None)
   next_ids=bonus.cuda()
   cad=CommonAttentionMetadata(query_start_loc=qstart,query_start_loc_cpu=torch.tensor(start,dtype=torch.int32),seq_lens=seq_gpu,num_reqs=batch,num_actual_tokens=total,max_query_len=max(context),max_seq_len=max(seq),block_table_tensor=table,slot_mapping=torch.zeros(total,dtype=torch.int64,device='cuda'),seq_lens_cpu_upper_bound=torch.tensor(seq,dtype=torch.int32))
   input_tensors=[positions,ids,hidden,qstart,seq_gpu,table_storage,next_ids]+([rejected] if rejected is not None else [])
   input_before=[x.cpu().clone() for x in input_tensors]
   for count in (1,2,3,7):
    for view in (False,True):
     os.environ['TIANSHU_DSPARK_SAMPLE_VIEW']='1' if view else '0'
     obj=DSparkProposer.__new__(DSparkProposer);obj.device=torch.device('cuda');obj.num_speculative_tokens=count;obj.backbone_block_size=7;obj.block_size=16;obj.parallel_drafting_token_id=75982;obj.dflash_causal=False
     stores=[]
     for name,size,dtype in [('input_ids',batch*7,torch.int64),('positions',batch*7,torch.int64),('_context_positions_buffer',total,torch.int64),('_context_slot_mapping_buffer',total,torch.int64),('_slot_mapping_buffer',batch*7,torch.int64)]:
      value,storage=guarded(size,dtype);setattr(obj,name,value);stores.append(storage)
     obj.arange=torch.arange(max(total,batch*7)+33,dtype=torch.int32,device='cuda');obj.token_arange_np=__import__('numpy').arange(batch+1,dtype='int32')
     cpu_rng=torch.random.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
     emitted,samples,new_cad=obj.set_inputs_first_pass(ids,next_ids,positions,hidden,None,cad,rejected)
     torch.cuda.synchronize()
     actual=dict(input_ids=obj.input_ids.cpu(),context_positions=obj._context_positions_buffer.cpu(),query_positions=obj.positions.cpu(),context_slots=obj._context_slot_mapping_buffer.cpu(),query_slots=obj._slot_mapping_buffer.cpu(),query_start_loc=new_cad.query_start_loc.cpu(),seq_lens=new_cad.seq_lens.cpu(),sample_indices=(torch.arange(emitted,dtype=torch.int32) if isinstance(samples,slice) else samples.cpu()))
     for key,val in actual.items():assert val.tolist()==expected[key],(case['name'],phase,count,view,key)
     assert emitted==batch*7 and obj.num_speculative_tokens==count and obj._dflash_hidden_states is hidden and obj._dflash_num_context==total
     assert new_cad.max_query_len==7 and new_cad.causal is False and new_cad.num_actual_tokens==batch*7
     assert new_cad.query_start_loc_cpu.tolist()==expected['query_start_loc']
     assert new_cad.seq_lens_cpu_upper_bound.tolist()==[x+7 for x in seq]
     assert new_cad.block_table_tensor is table and new_cad.slot_mapping.data_ptr()==obj._slot_mapping_buffer.data_ptr()
     assert all(bool((s[:11]==-177).all() and (s[-11:]==-177).all()) for s in stores)
     assert all(torch.equal(t.cpu(),b) for t,b in zip(input_tensors,input_before))
     assert torch.equal(cpu_rng,torch.random.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
     name=f"{case['name']}_phase{phase}_k{count}_view{int(view)}"
     archive=root/(name+'.pt');torch.save(dict(case=case,phase=phase,past=past,bonus=bonus,pages=pagedata,actual=actual),archive)
     state['cases'].append(dict(name=name,context_tokens=total,batch=batch,target_proposals=count,physical_draft_queries=batch*7,sample_view=view,phase=phase,passed=True,all_output_and_input_guards_preserved=True,RNG_unchanged=True,context_hidden_identity_preserved=True,tensor_archive_sha256=sha(archive)))
   save();print(json.dumps(dict(fixture=case['name'],phase=phase,completed=len(state['cases']))),flush=True)
 state.update(complete=True,phase='complete',actual_GPU_prepare_kernel_source=str(Path(utils.copy_and_expand_dflash_inputs_kernel.fn.__code__.co_filename)),elapsed_seconds=None)
except BaseException as error:
 state.update(phase='failed',error=repr(error));save();raise
save();print(json.dumps(dict(complete=True,cases=len(state['cases']),model_executed=False)),flush=True)
