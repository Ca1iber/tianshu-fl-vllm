# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,math,statistics,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261023_IR_RMS_A8_GPU_tianshu');PREP=json.loads((BASE/'CPU_PREPARE_GATE_32g.json').read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def decode(bits):return (np.asarray(bits,dtype=np.uint32)<<16).view(np.float32)
def encode(a):
 b=np.asarray(a,dtype=np.float32).view(np.uint32);return ((b+np.uint32(0x7fff)+((b>>16)&1))>>16).astype(np.uint16)
def expected_inputs(M,has,phase,bank):
 rng=np.random.default_rng(202610230700+100*bank+10*phase+int(has));x=rng.standard_normal((M,2048),dtype=np.float32);r=rng.standard_normal((M,2048),dtype=np.float32);scale=np.float32((1.,.0625,1.e-5)[phase]);x=encode(x*scale);r=encode(r*scale);x[0]=0;r[0]=0
 if phase==1 and M>1:r[1]=x[1]^np.uint16(0x8000)
 return x,r if has else None
assert not (BASE/'INDEPENDENT_CPU_GATE_32g.json').exists();result=json.loads((ROOT/'result.json').read_text());assert result['complete'] and len(result['cases'])==108 and len(result['timings'])==120 and result['Graph_return_objects_poison_replay_all108_passed'] and result['all_guards_passed'] and result['original_inputs_weights_preserved'] and result['GPU_RNG_preserved_per_case'] and result['actual_full_model_CUBIN_equivalence_not_asserted']
assert result['sources']==PREP['sources']
for path,h in result['sources'].items():assert sha(path)==h and sha(ROOT/'source_snapshot'/Path(path).relative_to('/'))==h
for name,entry in result['files'].items():assert Path(entry['path'])==ROOT/name and Path(entry['path']).stat().st_size==entry['bytes'] and sha(entry['path'])==entry['sha256']
def read(name,kind,shape):return np.fromfile(ROOT/name,dtype=kind).reshape(shape)
errors=[];elements=0;casekeys=set();u=2.**-24;gamma=4096*u/(1-4096*u)
for case in result['cases']:
 M=case['M'];has=case['has_residual'];phase=case['phase'];bank=case['bank'];key=(bank,M,has,phase);assert key not in casekeys;casekeys.add(key);assert case['actual_Graph_return_objects_poison_replay_exact'] and case['all_output_guards_passed'] and case['input_weight_RNG_preserved'];expected=PREP['selected_norms'][bank];assert case['bank_name']==expected['name'];x_bits=read(case['input_files']['x'],np.uint16,(M,2048));w_bits=read(case['input_files']['weight'],np.uint16,(2048,));x_expected,r_expected=expected_inputs(M,has,phase,bank);assert np.array_equal(x_bits,x_expected) and hashlib.sha256(w_bits.tobytes()).hexdigest()==expected['sha256'];x=decode(x_bits).astype(np.float64);weight=decode(w_bits).astype(np.float64);value=x
 if has:
  r_bits=read(case['input_files']['residual'],np.uint16,(M,2048));assert np.array_equal(r_bits,r_expected);value=x+decode(r_bits).astype(np.float64)
 variance=np.mean(value*value,axis=1,keepdims=True);ideal=value/np.sqrt(variance+1.e-6);ideal_product=ideal*weight;expected_mid=decode(encode(ideal)).astype(np.float64);ideal_IR=expected_mid*weight
 def ulp(a):
  absolute=np.abs(a);out=np.zeros_like(absolute);nz=absolute>0;out[nz]=np.exp2(np.maximum(np.floor(np.log2(absolute[nz])),-126)-7);return out
 bound=np.abs(ideal_product)*(8*gamma+32*np.finfo(np.float32).eps)+2*ulp(ideal)*np.abs(weight)+ulp(ideal_product)
 group_data={}
 for group in ('control','candidate'):
  outputs=case[group];y_bits=read(outputs['norm'],np.uint16,(M,2048));y=decode(y_bits);assert np.isfinite(y).all();error=np.abs(y.astype(np.float64)-ideal_IR);assert np.all(error<=bound),dict(case=key,group=group,max_error=float(error.max()),bound_violation=float(np.max(error-bound)))
  if has:assert np.array_equal(read(outputs['residual'],np.uint16,(M,2048)),encode(value))
  scale=np.maximum(np.max(np.abs(y),axis=1),np.float32(1.e-8))/np.float32(127.);scale=scale.astype(np.float32);quant=np.clip(np.rint(np.divide(y,scale[:,None],dtype=np.float32)),-127,127).astype(np.int8);got_q=read(outputs['q'],np.int8,(M,2048));got_s=read(outputs['scale'],np.float32,(M,));assert np.array_equal(got_q,quant) and np.array_equal(got_s.view(np.uint32),scale.view(np.uint32));assert np.isfinite(got_s).all() and (got_s>0).all()
  for field,name in outputs.items():assert (ROOT/name).read_bytes()==(ROOT/case[group+'_graph'][field]).read_bytes(),(key,group,field,'actualGraphdump differs')
  group_data[group]=dict(y=y_bits,q=got_q,s=got_s);errors.append(dict(bank=bank,M=M,has_residual=has,phase=phase,group=group,full_K_cells=M*2048,max_abs_error=float(error.max()),max_bound_violation=float(np.max(error-bound)),relative_L2=float(np.linalg.norm(y.astype(np.float64)-ideal_IR)/max(np.linalg.norm(ideal_IR),1.e-30)),CPU_A8_and_scale_exact=True,actual_Graph_raw_bytes_exact=True))
 diffs=dict(norm_BF16_different=int(np.count_nonzero(group_data['control']['y']!=group_data['candidate']['y'])),q_different=int(np.count_nonzero(group_data['control']['q']!=group_data['candidate']['q'])),scale_different=int(np.count_nonzero(group_data['control']['s'].view(np.uint32)!=group_data['candidate']['s'].view(np.uint32))))
 errors[-1].update(diffs);elements+=M*2048
assert len(casekeys)==108 and elements==14745600
summary=[]
for M in (8,64,128):
 for has in (False,True):
  for execution in ('eager','graph'):
   med={}
   for kind in ('compiled_IR_then_A8','fused_IR_A8'):
    rows=[r for r in result['timings'] if r['M']==M and r['has_residual']==has and r['execution']==execution and r['kind']==kind];assert len(rows)==5 and [r['round'] for r in rows]==[1,2,3,4,5] and all(math.isfinite(r['ms']) and r['ms']>0 and r['excluded']==(r['round']==1) for r in rows);med[kind]=statistics.median(r['ms'] for r in rows[1:])
   summary.append(dict(M=M,has_residual=has,execution=execution,control_ms=med['compiled_IR_then_A8'],candidate_ms=med['fused_IR_A8'],latency_reduction_percent=100*(1-med['fused_IR_A8']/med['compiled_IR_then_A8'])))
for filename,rows in (('KEY_ERRORS_32g.csv',errors),('KEY_TIMINGS_32g.csv',result['timings']),('KEY_SUMMARY_32g.csv',summary)):
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with (BASE/filename).open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
checkpoint=sha('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');assert checkpoint==PREP['checkpoint_sha256']==result['model_checkpoint_sha256'];primary=[r for r in summary if r['M']==64 and r['execution']=='graph'];assert len(primary)==2;lead=all(r['latency_reduction_percent']>3 for r in primary);gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,full_K_FP64_cells=elements,numeric_cases=108,timing_rows=120,all_inputs_regenerated_from_fixed_seeds=True,all_real_norm_weights_checkpoint_exact=True,all_full_K_norms_within_predeclared_FP32_BF16_bound=True,both_groups_A8_and_scales_independently_bit_exact=True,actual_Graph_raw_return_objects_full_bytes_rechecked=True,input_weight_RNG_and_guards_passed=True,compiled_candidate_kernels=result['candidate_compiled'],standalone_compiled_IR_baseline_only=True,actual_full_model_CUBIN_equivalence_not_asserted=True,fresh_model_quality_required=True,component_lead=lead,primary_M64_graph=primary,summary=summary,checkpoint_sha256=checkpoint,result_sha256=sha(ROOT/'result.json'),verifier_sha256=sha(__file__),official_benchmark=False,adopted=False);(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps(gate))
