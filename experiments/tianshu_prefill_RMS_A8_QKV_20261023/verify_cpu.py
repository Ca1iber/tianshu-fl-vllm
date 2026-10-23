# SPDX-License-Identifier: Apache-2.0
"""Independent CPU full-norm, A8, sampled INT64 projection and timing audit."""
import argparse,csv,hashlib,json,math,statistics,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;KINDS=('compiled_IR_A8_native_QKV','fused_IR_A8_native_QKV')
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def decode(bits):return (np.asarray(bits,dtype=np.uint32)<<16).view(np.float32)
def encode(values):
 bits=np.asarray(values,dtype=np.float32).view(np.uint32);return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
def inputs(M,layer,phase,has):
 seed=202610231400+100*layer+10*phase+int(has);rng=np.random.default_rng(seed);x=rng.standard_normal((M,2048),dtype=np.float32);r=rng.standard_normal((M,2048),dtype=np.float32);factor=np.float32((1.,.0625,1e-5)[phase]);x=encode(x*factor);r=encode(r*factor);x[0]=0;r[0]=0
 if has and phase==1:r[1]=x[1]^np.uint16(0x8000)
 return x,r if has else None,seed

def indices(index,M):
 rng=np.random.default_rng(202610231700+index);rows=[0,1,*sorted(rng.choice(np.arange(2,M-1),5,replace=False).tolist()),M-1];cols=[0,1,*sorted(rng.choice(np.arange(2,2559),13,replace=False).tolist()),2559];return rows,cols

def ulp(values):
 absolute=np.abs(values);out=np.zeros_like(absolute);nonzero=absolute>0;out[nonzero]=np.exp2(np.maximum(np.floor(np.log2(absolute[nonzero])),-126)-7);return out

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not (BASE/'INDEPENDENT_CPU_GATE_32g.json').exists();result=json.loads((root/'result.json').read_text());prep=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text());assert result['complete'] and result['phase']=='complete' and len(result['cases'])==18 and len(result['timings'])==180 and len(result['timing_graphs'])==24 and result['all_actual_Graph_outputs_full_bytes_checked'] and result['input_weight_RNG_and_guards_passed'] and result['no_full_model_CUBIN_equivalence_claim'] and result['fresh_model_quality_required'] and result['operator_calls_are_host_invocations_not_graph_replay_counts'] and result['numerical_kernel_unchanged']
 assert 'No running processes found' in (root/'hardware_before.txt').read_text() and 'No running processes found' in (root/'hardware_after.txt').read_text()
 for path,digest in result['sources'].items():assert sha(path)==digest and sha(root/'source_snapshot'/Path(path).relative_to('/'))==digest
 for path,digest in prep['sources'].items():assert result['sources'][path]==digest
 for name,entry in result['files'].items():
  path=root/name;assert path.is_relative_to(root) and Path(entry['path'])==path and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
  if entry['hardlink_to'] is not None:
   target=root/entry['hardlink_to'];assert target.is_relative_to(root) and path.stat().st_ino==target.stat().st_ino and path.stat().st_dev==target.stat().st_dev
 def read(name,dtype,shape):
  entry=result['files'][name];assert entry['shape']==list(shape);return np.fromfile(root/name,dtype=dtype).reshape(shape)
 assert [bank['layer'] for bank in result['banks']]==[0,20,41];banks={};compiled=result['compiled']
 for bank,expected in zip(result['banks'],prep['selected_banks']):
  layer=bank['layer'];assert layer==expected['layer'] and bank['has_residual']==expected['has_residual'] and bank['original_packs_exact'] and bank['original_BF16_sha256']==expected['original_BF16_sha256'];nt=read(bank['packed_NT'],np.int8,(2560,2048));weight_scale=read(bank['scale'],np.float32,(2560,));weight_bits=read(bank['norm'],np.uint16,(2048,));assert hashlib.sha256(nt.tobytes()).hexdigest()==expected['packed_NT_sha256'] and hashlib.sha256(weight_scale.tobytes()).hexdigest()==expected['scale_sha256'] and hashlib.sha256(weight_bits.tobytes()).hexdigest()==expected['norm_sha256'] and sha(expected['norm_path'])==expected['norm_sha256'];assert np.isfinite(weight_scale).all() and (weight_scale>0).all();banks[layer]=dict(nt=nt,scale=weight_scale,norm=decode(weight_bits).astype(np.float64))
 for M in (512,4096):
  for has in (False,True):assert compiled[f'fused_norm_M{M}_residual_{has}']['binary_sha256']==prep['expected_norm_binaries'][str(has)]
 for M in (512,4096):assert compiled[f'QKV_dequant_M{M}']['binary_sha256']==prep['expected_QKV_dequant_binaries'][str(M)]
 seen=set();errors=[];full_norm_cells=projection_cells=0;u=2.**-24;gamma=4096*u/(1-4096*u)
 for index,case in enumerate(result['cases']):
  layer=case['layer'];M=case['M'];has=case['has_residual'];phase=case['phase'];key=(layer,M,phase);assert case['case']==index and key not in seen and layer in (0,20,41) and M in (512,4096) and phase in (0,1,2) and has==(layer!=0) and case['actual_Graph_return_objects_full_bytes_poison_replay_exact'] and case['input_weights_RNG_and_all_guardable_outputs_passed'];seen.add(key);xb,rb,seed=inputs(M,layer,phase,has);assert case['input_seed']==seed;actual_x=read(case['input_files']['x'],np.uint16,(M,2048));assert np.array_equal(actual_x,xb);value=decode(actual_x).astype(np.float64)
  if has:
   actual_r=read(case['input_files']['residual'],np.uint16,(M,2048));assert np.array_equal(actual_r,rb);value=value+decode(actual_r).astype(np.float64)
  weight=banks[layer]['norm'];variance=np.mean(value*value,axis=1,keepdims=True);ideal=value/np.sqrt(variance+1e-6);ideal_product=ideal*weight;ideal_IR=decode(encode(ideal)).astype(np.float64)*weight;bound=np.abs(ideal_product)*(8*gamma+32*np.finfo(np.float32).eps)+2*ulp(ideal)*np.abs(weight)+ulp(ideal_product);sample_rows,sample_cols=indices(index,M);assert case['projection_sample_rows']==sample_rows and case['projection_sample_columns']==sample_cols;ix=np.ix_(sample_rows,sample_cols)
  for group in ('control','candidate'):
   output=case[group];assert set(output)==({'norm','residual','q','scale','projection'} if has else {'norm','q','scale','projection'});yb=read(output['norm'],np.uint16,(M,2048));y=decode(yb);error=np.abs(y.astype(np.float64)-ideal_IR);assert np.isfinite(y).all() and np.all(error<=bound),(key,group,float(np.max(error-bound)))
   if has:assert np.array_equal(read(output['residual'],np.uint16,(M,2048)),encode(value))
   scale=(np.maximum(np.max(np.abs(y),axis=1),np.float32(1e-8))/np.float32(127)).astype(np.float32);quant=np.clip(np.rint(np.divide(y,scale[:,None],dtype=np.float32)),-127,127).astype(np.int8);actual_q=read(output['q'],np.int8,(M,2048));actual_scale=read(output['scale'],np.float32,(M,));assert np.array_equal(actual_q,quant) and np.array_equal(actual_scale.view(np.uint32),scale.view(np.uint32)) and np.isfinite(actual_scale).all() and (actual_scale>0).all()
   integer_reference=actual_q[sample_rows].astype(np.int64)@banks[layer]['nt'][sample_cols].astype(np.int64).T;integer=read(case['integer_samples'][group]['eager'],np.int32,(8,16));assert np.array_equal(integer.astype(np.int64),integer_reference);scaled=np.multiply(integer_reference.astype(np.float32),actual_scale[sample_rows,None],dtype=np.float32);scaled=np.multiply(scaled,banks[layer]['scale'][None,sample_cols],dtype=np.float32);projection=read(output['projection'],np.uint16,(M,2560));assert np.array_equal(projection[ix],encode(scaled));projection_cells+=128
   for field,name in output.items():assert (root/name).read_bytes()==(root/case[group+'_graph'][field]).read_bytes()
   assert (root/case['integer_samples'][group]['eager']).read_bytes()==(root/case['integer_samples'][group]['graph']).read_bytes();integer_meta=case['full_integer_sha256'][group];assert integer_meta['eager']==integer_meta['graph'] and len(integer_meta['eager'])==64 and integer_meta['shape']==[M,2560] and integer_meta['full_GPU_bytes_compared']
   errors.append(dict(layer=layer,M=M,has_residual=has,phase=phase,group=group,full_K_norm_cells=M*2048,max_norm_abs_error=float(error.max()),max_norm_bound_violation=float(np.max(error-bound)),CPU_A8_and_scale_exact=True,full_K_INT64_projection_cells=128,INT32_and_BF16_projection_samples_exact=True,actual_graph_all_saved_output_bytes_exact=True,actual_graph_full_INT32_checked_on_GPU=True))
  full_norm_cells+=M*2048
 assert len(seen)==18 and full_norm_cells==84934656 and projection_cells==4608
 graphs=set()
 for graph in result['timing_graphs']:
  index=graph['case'];case=result['cases'][index];kind=graph['kind'];length=graph['chain_length'];group='control' if kind==KINDS[0] else 'candidate';key=(index,kind,length);assert key not in graphs and case['phase']==0 and kind in KINDS and length in (4,16) and graph['actual_graph_captured_full_chain_calls']==length and graph['actual_Graph_return_objects_poison_replay_exact'] and graph['integer_full_bytes_compared'] and (graph['layer'],graph['M'],graph['has_residual'])==(case['layer'],case['M'],case['has_residual']);graphs.add(key);expected={field:result['files'][name]['sha256'] for field,name in case[group].items()};expected['integer']=case['full_integer_sha256'][group]['eager'];assert graph['outputs_sha256']==expected
 assert len(graphs)==24
 summary=[];medians={};timing_keys=set()
 for layer in (0,20,41):
  for M in (512,4096):
   case=next(c for c in result['cases'] if (c['layer'],c['M'],c['phase'])==(layer,M,0))
   for execution,length in (('eager',1),('graph',4),('graph',16)):
    median={}
    for kind in KINDS:
     rows=[row for row in result['timings'] if (row['layer'],row['M'],row['execution'],row['chain_length'],row['kind'])==(layer,M,execution,length,kind)];assert len(rows)==5 and [r['round'] for r in rows]==[1,2,3,4,5]
     for row in rows:
      key=(layer,M,execution,length,kind,row['round']);assert key not in timing_keys;timing_keys.add(key);order=KINDS if row['round']%2 else KINDS[::-1];assert row['case']==case['case'] and row['has_residual']==(layer!=0) and row['excluded']==(row['round']==1) and row['warm_replays']==10 and row['timed_replays']==20 and row['order_within_round']==order.index(kind) and math.isfinite(row['ms_per_complete_chain']) and row['ms_per_complete_chain']>0 and row['ms_per_complete_chain']==row['total_event_ms']/(20*length)
     median[kind]=statistics.median(row['ms_per_complete_chain'] for row in rows[1:]);medians[layer,M,execution,length,kind]=median[kind]
    summary.append(dict(layer=layer,M=M,has_residual=layer!=0,execution=execution,chain_length=length,control_ms=median[KINDS[0]],candidate_ms=median[KINDS[1]],latency_reduction_percent=100*(1-median[KINDS[1]]/median[KINDS[0]])))
 assert len(timing_keys)==180 and len(summary)==18
 sensitivity=[]
 for layer in (0,20,41):
  for kind in KINDS:
   a=medians[layer,4096,'graph',4,kind];b=medians[layer,4096,'graph',16,kind];sensitivity.append(dict(layer=layer,kind=kind,chain4_vs16_percent=100*abs(a/b-1)))
 primary=[row for row in summary if row['M']==4096 and (row['execution']=='eager' or row['chain_length']==16)];assert len(primary)==6;stable=all(row['chain4_vs16_percent']<=10 for row in sensitivity);lead=stable and all(row['latency_reduction_percent']>3 for row in primary)
 for filename,rows in (('KEY_ERRORS_32g.csv',errors),('KEY_TIMINGS_32g.csv',result['timings']),('KEY_SUMMARY_32g.csv',summary)):
  fields=list(dict.fromkeys(key for row in rows for key in row))
  with (BASE/filename).open('w',newline='') as stream:writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
 checkpoint=sha('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');assert checkpoint==prep['checkpoint_sha256']==result['model_checkpoint_sha256']=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c';assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
 gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,component_lead=lead,numeric_cases=18,full_K_FP64_norm_cells=full_norm_cells,both_arms_all_norm_cells_within_predeclared_bound=True,both_arms_all_A8_and_scales_independently_exact=True,full_K_INT64_projection_reference_cells=projection_cells,projection_reference_is_sampled_not_full_matrix=True,all_projection_samples_INT32_and_final_BF16_exact=True,numeric36_actual_graph_outputs_full_bytes_rechecked=True,timing24_actual_graph_outputs_full_bytes_rechecked=True,all_raw_file_paths_SHA_and_exact_graph_file_links_verified=True,input_weight_RNG_and_guardable_outputs_passed=True,all180_timing_rows_verified=True,first_round_excluded=True,M4096_chain4_vs16_stable_within10percent=stable,M4096_complete_chain_eager_and_graph_primary=primary,chain_sensitivity=sensitivity,summary=summary,compiled=result['compiled'],all_sources_and_snapshots_SHA_verified=True,checkpoint_sha256=checkpoint,standalone_compiled_native_IR_baseline_only=True,no_full_model_CUBIN_equivalence_claim=True,fresh_model_C8_C64_quality_required=True,original_checkpoint_and_structure_unchanged=True,original_three_QKV_weight_pack_signatures_exact=True,result_sha256=sha(root/'result.json'),verifier_sha256=sha(__file__),official_benchmark=False,adopted=False)
 (BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))

if __name__=='__main__':main()
