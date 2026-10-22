# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261022_native_weight_only_contract_GPU_tianshu')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bf16(v):
 b=np.asarray(v,np.float32).view(np.uint32);return ((b+32767+((b>>16)&1))>>16).astype(np.uint16)
def f32(w):return (np.asarray(w,np.uint16).astype(np.uint32)<<16).view(np.float32)
def main():
 assert not (BASE/'CPU_GATE_32g.json').exists();r=json.loads((ROOT/'result.json').read_text());assert r['complete'] and r['phase']=='complete' and r['accepted_cases']==12 and r['contract_rejections']==0 and len(r['cases'])==12 and r['native_provider_and_source_hashes_unchanged'];rows=[];outputs={};cells=0
 for p,h in r['sources'].items():
  assert sha(p)==h
  if Path(p).name in ('PLAN_32g.md','probe_gpu.py'):assert sha(ROOT/'source_snapshot'/Path(p).name)==h
 seen=set()
 for case in r['cases']:
  m=case['M'];layout=case['layout'];dtype=case['scales_dtype'];assert m in (8,64,128) and layout in ('NT_contiguous','NN_transposed_view') and dtype in ('FP32','BF16') and (m,layout,dtype) not in seen and case['accepted'] and len(case['phases'])==2;seen.add((m,layout,dtype))
  for phase,proof in enumerate(case['phases']):
   assert proof['phase']==phase and proof['graph_eager_and_guards_passed'] and sha(ROOT/proof['artifact'])==proof['sha256'];rng=np.random.default_rng(202610236000+m+phase);xb=bf16(rng.standard_normal((m,256),dtype=np.float32));xb[0]=0;wc=rng.integers(-127,128,(256,256),dtype=np.int16).astype(np.int8);sv=(np.arange(256,dtype=np.float32)%31+1)/(np.float32(257) if phase==0 else np.float32(263));sv=f32(bf16(sv)) if dtype=='BF16' else sv
   with np.load(ROOT/proof['artifact']) as a:
    assert np.array_equal(a['X_words'],xb) and np.array_equal(a['W_codes'],wc) and np.array_equal(a['scales'],sv);out=a['O_words'];assert out.shape==(m,256) and ahash(out)==proof['output_sha256'];h=f32(xb).astype(np.float64);w=wc.astype(np.float64);s=sv.astype(np.float64);ideal=(h@w.T)*s[None,:];absolute=(abs(h)@abs(w).T)*s[None,:];bits=out;exponent=((bits&0x7f80)>>7).astype(np.int32);ulp=np.exp2(np.where(exponent==0,-133,exponent-134).astype(np.float64));eps=2.**-24;gamma=516*eps/(1-516*eps);actual=f32(out).astype(np.float64);bound=gamma*absolute+.5*ulp+1e-10;assert np.isfinite(actual).all() and not actual[0].any() and np.all(abs(actual-ideal)<=bound);key=(m,dtype,phase)
    if key in outputs:assert np.array_equal(outputs[key],out)
    else:outputs[key]=out.copy()
    cells+=out.size;rows.append(dict(M=m,layout=layout,scales_dtype=dtype,phase=phase,output_cells=out.size,full_post_scale_CPU_FP64_bound_passed=True,relative_L2=float(np.linalg.norm(actual-ideal)/np.linalg.norm(ideal))))
 assert cells==409600 and len(rows)==24 and not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
 with (BASE/'KEY_CASES_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,accepted_BF16_activation_cases=12,artifacts_checked=24,full_independent_CPU_FP64_output_cells=cells,all_fixtures_reconstructed=True,all_output_cells_within_original_post_scale_bound=True,NT_and_NN_layout_outputs_identical=True,native_provider_and_source_hashes_verified=True,original_GPU_graph_eager_and_immutability_guards_passed=True,CPU_did_not_rerun_GPU_guards=True,mathematical_bound_not_bitexact_reference=True,real_weight_speed_unknown=True,model_executed=False,official_benchmark=False,adopted=False,FP32_wrapper_condition_not_met_not_executed=True,raw_root=str(ROOT),raw_result_sha256=sha(ROOT/'result.json'),provider_sources=r['sources'],verifier_sha256=sha(__file__),plan_sha256=sha(BASE/'PLAN_32g.md'),key_cases_sha256=sha(BASE/'KEY_CASES_32g.csv'));(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='provider_sources'}))
if __name__=='__main__':main()
