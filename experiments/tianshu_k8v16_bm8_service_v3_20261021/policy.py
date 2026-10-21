# SPDX-License-Identifier: Apache-2.0
"""M64-only child selector; actual JIT source/grid override is recorded explicitly."""
import atexit,functools,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
PAIR_SHA='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def install(worker,mode,directory):
 assert mode=='candidate'
 import torch
 import vllm.v1.attention.backends.triton_attn as backend
 from vllm_fl.worker.model_runner import ModelRunnerFL
 cfg=worker.vllm_config;assert cfg.speculative_config is None and cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64 and cfg.scheduler_config.async_scheduling
 selection=json.loads((BASE/'selected_plan.json').read_text());gate=json.loads(Path(selection['component_gate']).read_text())
 assert sha(selection['component_gate'])==selection['component_gate_sha256'] and gate['complete'] and gate['all_candidate_output_tensors_bitwise_equal_to_current'] and gate['full_output_elements_rechecked']==8044544
 assert all(sha(p)==h for p,h in gate['frozen_sources'].items()) and sha(selection['kernel_source'])==selection['kernel_sha256']
 assert sha(selection['binary_signature_source'])==selection['binary_signature_sha256'] and sha(selection['binary_code_gate'])==selection['binary_code_gate_sha256']
 bs=importlib.util.spec_from_file_location('_runtime_strict_ELF_sections',selection['binary_signature_source']);binary_checker=importlib.util.module_from_spec(bs);bs.loader.exec_module(binary_checker)
 producer=backend._soft8_pipeline_installed['producer'];pair_router=producer.kernel;old=pair_router.candidate
 assert sha(inspect.getsourcefile(old.fn))==PAIR_SHA and pair_router.mode=='candidate'
 assert Path(inspect.getsourcefile(type(pair_router).__getitem__)).resolve()==BASE.parent/'tianshu_k8v16_pair_service_20261020/policy.py'
 spec=importlib.util.spec_from_file_location('_live_BM8_direct_attention',selection['kernel_source']);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);trace=sys.modules['_tianshu_long_segments_actual_graph_trace'].STATE
 record=dict(installed=True,mode=mode,completed=False,new_path_enabled_during_capture=True,selected_sequences=[64],IS_3D_only=True,no_new_packed_weights=True,RNG_not_read_or_modified=True,other_batches_and_2D_delegate_original=True,old_pair_record_scope='Base pair selector and parameters BEFORE this child mapping; final source/grid/binary are in this new record',unchanged_writer_cache_format_softmax_reducer_and_sampling=True,policy_source=str(BASE/'policy.py'),policy_sha256=sha(__file__),selection_source=str(BASE/'selected_plan.json'),selection_sha256=sha(BASE/'selected_plan.json'),kernel_source=selection['kernel_source'],kernel_sha256=selection['kernel_sha256'],base_pair_source_sha256=PAIR_SHA,routes={},Python_calls_not_Graph_replay_counts=True)
 def save():
  p=directory/f'binding_{os.getpid()}.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(record,indent=2)+'\n');t.replace(p)
 class Selected:
  fn=old.fn
  def __getitem__(self,grid):
   def launch(*args,**kwargs):
    assert not args
    selected=bool(kwargs['IS_3D']) and kwargs['num_seqs']==64
    actual_grid=grid;actual=dict(kwargs)
    if selected:
     assert kwargs['BLOCK_M']==16 and kwargs['BLOCK_Q']==2 and kwargs['TILE_SIZE']==16 and kwargs['NUM_SEGMENTS_PER_SEQ']==4
     assert kwargs['HEAD_SIZE']==kwargs['HEAD_SIZE_PADDED']==128 and kwargs['num_query_heads']==16 and kwargs['num_queries_per_kv']==8 and kwargs['KV_QUANT_MODE']==0
     assert not kwargs['USE_TD'] and not kwargs['USE_TD_QO'] and kwargs['query_ptr'].dtype==kwargs['value_cache_ptr'].dtype==torch.bfloat16 and kwargs['key_cache_ptr'].dtype==torch.uint8
     actual.update(BLOCK_M=8,BLOCK_Q=1);actual_grid=(64,grid[1],grid[2]);compiled=module.kernel_unified_attention[actual_grid](**actual)
    else:compiled=old[grid](**kwargs)
    capturing=torch.cuda.is_current_stream_capturing();key=str((trace.get('active_graph'),capturing,bool(kwargs['IS_3D']),kwargs['num_seqs'],selected))
    row=record['routes'].setdefault(key,dict(graph_key=trace.get('active_graph'),actually_inside_capture=capturing,sequences=kwargs['num_seqs'],IS_3D=bool(kwargs['IS_3D']),selected=selected,original_grid=list(grid),actual_grid=list(actual_grid),BLOCK_M=actual['BLOCK_M'],BLOCK_Q=actual['BLOCK_Q'],TILE_SIZE=actual['TILE_SIZE'],segments=actual['NUM_SEGMENTS_PER_SEQ'],actual_source_sha256=selection['kernel_sha256'] if selected else PAIR_SHA,calls=0))
    if selected and 'binary_sha256' not in row:
     cache_hash=compiled.hash;binary=hashlib.sha256(compiled.kernel).hexdigest();expected=selection['timed_component_kernels'].get(cache_hash)
     row.update(cache_hash=cache_hash,binary_sha256=binary,n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared,num_warps=compiled.metadata.num_warps,num_stages=compiled.metadata.num_stages)
     sig=binary_checker.signature(compiled.kernel);row.update(non_debug_signature_sha256=sig['non_debug_signature_sha256'],ignored_debug_info=sig['ignored_debug_info'],full_binary_equal_to_component=(expected is not None and binary==expected['binary_sha256']))
     save();assert expected is not None and sig['canonical']==expected['non_debug_canonical'] and sig['non_debug_signature_sha256']==expected['non_debug_signature_sha256'],dict(actual=cache_hash,allowed=list(selection['timed_component_kernels']))
     assert compiled.n_regs==expected['n_regs'] and compiled.n_spills==expected['n_spills']==0 and compiled.metadata.shared==expected['shared'] and compiled.metadata.num_warps==compiled.metadata.num_stages==1
     row['timed_component_machine_code_exact_match']=True
    row['calls']+=1
    if row['calls']==1 or not capturing and row['calls']&(row['calls']-1)==0:save()
    return compiled
   return launch
 pair_router.candidate=Selected();original_capture=ModelRunnerFL.capture_model
 @functools.wraps(original_capture)
 def capture(owner,*args,**kwargs):
  value=original_capture(owner,*args,**kwargs)
  rows=[r for r in record['routes'].values() if r['selected'] and r['actually_inside_capture']]
  assert rows and sum(r['calls'] for r in rows)>=42 and all(r['timed_component_machine_code_exact_match'] for r in rows)
  record.update(completed=True,selected_real_capture_calls=sum(r['calls'] for r in rows));save();return value
 ModelRunnerFL.capture_model=capture;atexit.register(save);save()
