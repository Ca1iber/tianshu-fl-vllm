# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/workspace/benchmark-results/20261022_LM_pad64_GPU_tianshu');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bf(w):return (np.asarray(w,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def main():
 import torch
 from safetensors import safe_open
 assert not torch.cuda.is_initialized() and not (BASE/'INDEPENDENT_CPU_GATE_32g.json').exists();state=json.loads((ROOT/'result.json').read_text());assert state['complete'] and len(state['phases'])==15 and len(state['rounds'])==70
 for p,h in state['frozen_sources'].items():assert sha(p)==h and sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))==h
 with safe_open(str(MODEL),framework='pt',device='cpu') as cp:weight=cp.get_tensor('lm_head.weight').contiguous();words=weight.view(torch.uint16).numpy().copy()
 wf=bf(words);ws=np.maximum(np.abs(wf).max(1),np.float32(1.e-8))/np.float32(127.);wq=np.clip(np.rint(wf/ws[:,None]),-127,127).astype(np.int8);assert hashlib.sha256(words.tobytes()).hexdigest()==state['weight_signatures']['original_BF16_sha256'] and hashlib.sha256(wq.tobytes()).hexdigest()==state['weight_signatures']['packed_sha256'] and hashlib.sha256(ws.tobytes()).hexdigest()==state['weight_signatures']['scales_sha256'];cells=0;errors=[]
 for phase in state['phases']:
  path=ROOT/phase['artifact'];assert sha(path)==phase['sha256'];d=np.load(path);m=phase['M'];sample=np.random.default_rng(phase['seed']).standard_normal((m,2048),dtype=np.float32);b=sample.view(np.uint32);aw=((b+32767+((b>>16)&1))>>16).astype(np.uint16);aw[0]=0;a=np.zeros((64,2048),np.float32);a[:m]=bf(aw);asc=np.maximum(np.abs(a).max(1),np.float32(1.e-8))/np.float32(127.);q=np.clip(np.rint(a/asc[:,None]),-127,127).astype(np.int8);assert np.array_equal(q,d['Aq']) and np.array_equal(asc,d['As']);r,c=d['rows'],d['columns'];integer=q[r].astype(np.int64)@(wq[c].astype(np.int64)*phase['sign']).T;assert np.array_equal(integer,d['integer_selected']);value=(integer.astype(np.float32)*asc[r,None])*(ws[c][None,:]*np.float32(phase['factor']));v=value.view(np.uint32);rounded=((v+32767+((v>>16)&1))>>16).astype(np.uint16);assert np.array_equal(bf(rounded),d['output_FP32_selected']);cells+=integer.size
  exact=a[r].astype(np.float64)@(wf[c].astype(np.float64)*(phase['sign']*phase['factor'])).T
  if m<64:
   absdot=np.abs(a[r].astype(np.float64))@np.abs(wf[c].astype(np.float64)*phase['factor']).T;u=2**-24;gamma=4096*u/(1-4096*u);exponent=np.frexp(np.abs(exact))[1];ulp=np.maximum(np.ldexp(np.ones_like(exact),exponent-8),2**-133);assert np.all(np.abs(d['baseline_FP32_selected']-exact)<=gamma*absdot+ulp)
  errors.append(dict(M=m,phase=phase['phase'],relative_L2=float(np.linalg.norm(d['output_FP32_selected']-exact)/np.linalg.norm(exact))))
 summary={};timing_rows=[]
 for m in (16,32,48,63,64):
  summary[str(m)]={}
  for mode in ('eager','graph'):
   rounds=[r for r in state['rounds'] if r['M']==m and r['mode']==mode];assert len(rounds)==7 and [r['round'] for r in rounds]==list(range(7));med={name:statistics.median(r['milliseconds'][name] for r in rounds[1:]) for name in ('actual_control','padded_fused')};summary[str(m)][mode]=dict(median_ms=med,speedup_percent=100*(med['actual_control']/med['padded_fused']-1))
   for r in rounds:
    for name,value in r['milliseconds'].items():timing_rows.append(dict(M=m,mode=mode,round=r['round'],excluded=r['excluded'],name=name,milliseconds=value))
 cutoff=next((cut for cut in (32,48,64) if all(summary[str(m)]['eager']['speedup_percent']>5 for m in (16,32,48,63,64) if m>=cut)),None)
 for name,rows in [('KEY_TIMINGS_32g.csv',timing_rows),('KEY_ERRORS_32g.csv',errors)]:
  with (BASE/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 assert sha(MODEL)==state['checkpoint_before']==state['checkpoint_after']=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c' and not torch.cuda.is_initialized();gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,CPU_full_K_INT64_BF16_FP32_cells=cells,all_original_pack_signatures_exact=True,all_source_snapshots_verified=True,full_GPU_all64_INT32_FP32_exact=True,M64_original_qualified_bits_exact=True,smaller_M_new_precision_requires_fresh_quality=True,compiled_kernels=state['compiled'],actual_vendor_traces=state['actual_vendor_traces'],summary=summary,suggested_cutoff=cutoff,dynamic_range_lead=(cutoff is not None and cutoff<64),model_not_executed=True,quality_not_tested=True,official_benchmark=False,adopted=False,raw_result_sha256=sha(ROOT/'result.json'),verifier_sha256=sha(__file__));(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
