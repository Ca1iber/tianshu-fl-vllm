# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,math,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261023_IR_RMS_A8_chain_GPU_tianshu')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert not (BASE/'INDEPENDENT_CPU_GATE_32g.json').exists()
prep=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text());r=json.loads((ROOT/'result.json').read_text());old=json.loads(Path(prep['prior_result']).read_text());gate=json.loads(Path(prep['prior_gate']).read_text());assert gate['complete'] and gate['numeric_cases']==108 and gate['full_K_FP64_cells']==14745600 and gate['actual_Graph_raw_return_objects_full_bytes_rechecked'] and gate['both_groups_A8_and_scales_independently_bit_exact'] and r['complete'] and r['sources']==prep['sources'] and r['candidate_compiled']==old['candidate_compiled']
for p,h in r['sources'].items():assert sha(p)==h and sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))==h
assert sha(prep['prior_result'])==gate['result_sha256']
for name,e in r['files'].items():assert Path(e['path'])==ROOT/name and Path(e['path']).stat().st_size==e['bytes'] and sha(e['path'])==e['sha256']
seen=set()
for graph in r['graphs']:
 key=(graph['M'],graph['has_residual'],graph['kind'],graph['chain_length']);assert key not in seen;seen.add(key);assert graph['actual_graph_captured_calls']==graph['chain_length'] and graph['poison_replay_exact_to_prior_complete_outputs'] and graph['input_weight_RNG_unchanged'];c=old['cases'][graph['prior_case']];assert (c['M'],c['has_residual'],c['bank'],c['phase'])==(graph['M'],graph['has_residual'],0,0);group='control' if graph['kind']=='compiled_IR_then_A8' else 'candidate';assert group==graph['prior_group'] and set(graph['saved_outputs'])==set(c[group])
 for field,name in graph['saved_outputs'].items():
  oldfile=old['files'][c[group][field]];assert sha(oldfile['path'])==oldfile['sha256'] and (ROOT/name).read_bytes()==Path(oldfile['path']).read_bytes()
assert len(seen)==36 and len(r['timings'])==180
summary=[];medians={}
for M in (8,64,128):
 for has in (False,True):
  for chain in (1,32,128):
   med={}
   for kind in ('compiled_IR_then_A8','fused_IR_A8'):
    rows=[x for x in r['timings'] if (x['M'],x['has_residual'],x['chain_length'],x['kind'])==(M,has,chain,kind)];assert len(rows)==5 and [x['round'] for x in rows]==[1,2,3,4,5]
    for row in rows:
     assert row['warm_replays']==10 and row['timed_replays']==20 and row['excluded']==(row['round']==1) and row['order_within_round']==(('compiled_IR_then_A8','fused_IR_A8') if row['round']%2 else ('fused_IR_A8','compiled_IR_then_A8')).index(kind)
     assert math.isfinite(row['ms_per_component']) and row['ms_per_component']>0 and row['ms_per_component']==row['total_graph_ms']/(20*chain)
    med[kind]=statistics.median(x['ms_per_component'] for x in rows[1:]);medians[M,has,chain,kind]=med[kind]
   summary.append(dict(M=M,has_residual=has,chain_length=chain,control_us=med['compiled_IR_then_A8']*1000,candidate_us=med['fused_IR_A8']*1000,latency_reduction_percent=100*(1-med['fused_IR_A8']/med['compiled_IR_then_A8'])))
primary=[x for x in summary if x['M']==64 and x['chain_length']==128];sensitivities=[]
for has in (False,True):
 for kind in ('compiled_IR_then_A8','fused_IR_A8'):
  a=medians[64,has,32,kind];b=medians[64,has,128,kind];sensitivities.append(dict(has_residual=has,kind=kind,chain32_vs128_percent=100*abs(a/b-1)))
stable=all(x['chain32_vs128_percent']<=10 for x in sensitivities);lead=stable and all(x['latency_reduction_percent']>3 for x in primary)
for filename,rows in (('KEY_TIMINGS_32g.csv',r['timings']),('KEY_SUMMARY_32g.csv',summary)):
 with (BASE/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
result=dict(complete=True,CPU_only=True,GPU_not_imported=True,measurement_method_changed=True,numerical_kernel_unchanged_and_CUBIN_exact=True,inherited_108_numeric_cases_verified=True,actual36_graph_outputs_full_bytes_exact_to_prior=True,all180_timing_rows_verified=True,first_round_excluded=True,chain32_vs128_stable_within10percent=stable,component_lead=lead,primary_M64_chain128=primary,chain_sensitivity=sensitivities,summary=summary,prior_numeric_gate_sha256=sha(prep['prior_gate']),prior_result_sha256=sha(prep['prior_result']),result_sha256=sha(ROOT/'result.json'),verifier_sha256=sha(__file__),no_full_model_CUBIN_equivalence_claim=True,fresh_model_quality_required=True,official_benchmark=False,adopted=False)
(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
