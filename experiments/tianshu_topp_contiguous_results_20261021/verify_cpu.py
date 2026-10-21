# SPDX-License-Identifier: Apache-2.0
"""Reconstruct every full FP32 GPU output from saved masks and bit-exact BF16 input."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((a.raw/'result.json').read_text());assert r['complete'] and r['phase']=='complete' and len(r['cases'])==12 and not r['model_executed'] and not r['official_benchmark']
assert r['numerical_JIT_source_unchanged'] and r['only_warp_count_varies']
for path,h in r['frozen_sources'].items():assert sha(path)==h and sha(a.raw/'source_snapshot'/Path(path).relative_to('/'))==h
import numpy as np
import torch
torch.set_num_threads(8)
def tensor_sha(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
labels=['native','warp2','warp4','warp8','warp16'];rows=[];full=0;elements=0;ideal_rows=[];qualified={label:True for label in labels}
for case in r['cases']:
 assert case["actual_logit_row_stride"]==case["vocab"] and case["layout"]=="contiguous_guard_rows"
 assert len(case['phases'])==3 and case['timed_input_RNG_and_guards_preserved']
 assert case['compiled']['native']['num_warps']==4 and case['compiled']['warp4']['binary_sha256']==case['compiled']['native']['binary_sha256']
 for label,meta in case['compiled'].items():
  assert meta['BLOCK_SIZE']==8192 and meta['BLOCK_SIZE_TRUNC']==4096 and meta['num_warps']==(4 if label=='native' else int(label[4:]))
 for phase in case['phases']:
  file=a.raw/phase['archive'];assert sha(file)==phase['sha256'];data=torch.load(file,map_location='cpu',weights_only=True)
  original=data['input_bf16'].float();batch,vocab=original.shape;native_out=None
  for label in labels:
   out=data['outputs'][label];assert out['retained_values_bitwise_original'] and out['guards_preserved'] and out['inputs_preserved'] and out['RNG_unchanged']
   keep=torch.from_numpy(np.unpackbits(out['keep_mask_packbits_little'].numpy(),bitorder='little',count=batch*vocab).copy()).bool().reshape(batch,vocab)
   actual=original.clone();actual[~keep]=-float('inf');assert tensor_sha(actual)==out['full_output_sha256']
   if label=='native':native_out=actual.clone()
   equal=torch.equal(actual.view(torch.int32),native_out.view(torch.int32));different=int((actual.view(torch.int32)!=native_out.view(torch.int32)).sum())
   assert equal==out['full_output_bitwise_native'] and different==out['different_elements']
   assert {k:v for k,v in out.items() if k!='keep_mask_packbits_little'}==phase['comparisons'][label]
   qualified[label]=qualified[label] and equal;full+=1;elements+=actual.numel()
  # Independent ideal nucleus diagnostic; preserve native differences instead of changing the contract.
  for i in range(batch):
   x=original[i].double();pval=float(data['p'][i]);finite=torch.isfinite(x)
   if bool(torch.isnan(x).any() or torch.isposinf(x).any()) or not bool(finite.any()):continue
   probs=torch.softmax(x,0);order=torch.argsort(x,descending=True,stable=True);cumulative=torch.cumsum(probs[order],0)
   if pval>=1:nkeep=int(finite.sum())
   else:nkeep=min(int(finite.sum()),max(1,int((cumulative<pval).sum())+1))
   ideal=torch.zeros(vocab,dtype=torch.bool);ideal[order[:nkeep]]=True
   native_keep=~torch.isneginf(native_out[i]);diff=int((ideal!=native_keep).sum());mass=float(probs[native_keep].sum())
   ideal_rows.append(dict(case=case['name'],phase=phase['phase'],row=i,p=pval,ideal_keep=nkeep,native_keep=int(native_keep.sum()),native_vs_ideal_mask_difference=diff,native_retained_mass=mass,original_finite_count=int(finite.sum())))
 for label in labels:
  times=[x['milliseconds'] for x in r['timing_rounds'] if x['case']==case['name'] and x['label']==label]
  assert len(times)==7 and all(t>0 for t in times);median=statistics.median(times);assert median==case['median_ms'][label]
  rows.append(dict(case=case['name'],batch=case['batch'],kind=case['kind'],label=label,median_ms=median,vs_native_filter_throughput_percent=100*(case['median_ms']['native']/median-1),all_three_full_outputs_bitwise_native=all(x['comparisons'][label]['full_output_bitwise_native'] for x in case['phases']),n_regs=case['compiled'][label]['n_regs'],n_spills=case['compiled'][label]['n_spills']))
assert full==180 and len(r['timing_rounds'])==420
proof=dict(complete=True,CPU_reverified=True,model_executed=False,official_benchmark=False,adopted=False,raw_root=str(a.raw),raw_result_sha256=sha(a.raw/'result.json'),source_count=len(r['frozen_sources']),frozen_sources=r['frozen_sources'],full_FP32_outputs_reconstructed_from_masks_and_original_values=full,full_FP32_output_values=elements,ideal_CPU_FP64_nucleus_rows=len(ideal_rows),ideal_diagnostic_differences_preserved=True,bitwise_native_all_cases_by_config=qualified,timing_trials=len(r['timing_rounds']),timing_scope=r['timing_scope'],verifier_sha256=sha(__file__),rows=rows)
(a.output/'GPU_GATE.json').write_text(json.dumps(proof,indent=2)+'\n');(a.output/'NATIVE_IDEAL_NUCLEUS_DIAGNOSTIC.json').write_text(json.dumps(ideal_rows,indent=2)+'\n')
with (a.output/'KEY_TIMINGS.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in proof.items() if k not in ('rows','frozen_sources')}))
