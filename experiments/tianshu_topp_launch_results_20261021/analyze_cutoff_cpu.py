# SPDX-License-Identifier: Apache-2.0
"""Characterize existing native/candidate nucleus differences; do not change GPU gates."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
import numpy as np
import torch
torch.set_num_threads(8)
r=json.loads((a.raw/'result.json').read_text());assert r['complete'] and r['phase']=='complete'
rows=[];summary={}
for case in r['cases']:
 for phase in case['phases']:
  data=torch.load(a.raw/phase['archive'],map_location='cpu',weights_only=True);x=data['input_bf16'].float();batch,vocab=x.shape
  masks={label:torch.from_numpy(np.unpackbits(out['keep_mask_packbits_little'].numpy(),bitorder='little',count=batch*vocab).copy()).bool().reshape(batch,vocab) for label,out in data['outputs'].items()}
  for i in range(batch):
   source=x[i].double();pval=float(data['p'][i]);finite=torch.isfinite(source)
   if bool(torch.isnan(source).any() or torch.isposinf(source).any()) or not bool(finite.any()):continue
   probs=source.softmax(0);order=torch.argsort(source,descending=True,stable=True);prefix=probs[order].cumsum(0)
   count=int(finite.sum()) if pval>=1 else min(int(finite.sum()),max(1,int((prefix<pval).sum())+1))
   ideal=torch.zeros(vocab,dtype=torch.bool);ideal[order[:count]]=True
   for label,mask in masks.items():
    keep=mask[i]&finite;removed=(~keep)&finite;mass=float(probs[keep].sum());num=int(keep.sum())
    minimum=float(probs[keep].min()) if num else 0.;rank_valid=(not bool(removed.any()) or (num>0 and float(source[keep].min())>=float(source[removed].max())))
    nucleus_valid=(num>0 and (mass>=pval or pval>=1 and num==int(finite.sum())))
    minimal_count=(num==1 if pval<=0 else (num==int(finite.sum()) if pval>=1 else mass-minimum<pval))
    native_diff=int((mask[i]!=masks['native'][i]).sum());ideal_diff=int((keep!=ideal).sum())
    changed=torch.nonzero(mask[i]!=masks['native'][i]).flatten()
    row=dict(case=case['name'],phase=phase['phase'],row=i,label=label,p=pval,kept=num,ideal_keep=count,kept_mass=mass,boundary_probability=minimum,mass_minus_p=mass-pval,rank_order_valid=rank_valid,nucleus_mass_valid_CPU_FP64=nucleus_valid,minimal_count_CPU_FP64=minimal_count,mask_difference_native=native_diff,mask_difference_stable_ideal=ideal_diff,changed_indices_first16=changed[:16].tolist(),changed_logit_values_first16=source[changed[:16]].tolist())
    rows.append(row);key=f"{label}_p{'095' if abs(pval-.95)<1e-7 else 'other'}"
    agg=summary.setdefault(key,dict(rows=0,rows_different_native=0,different_native_elements=0,rows_different_stable_ideal=0,rank_inversions=0,nucleus_mass_below_p=0,nonminimal_count=0,max_mass_shortfall=0.,max_mask_difference_native=0))
    agg['rows']+=1;agg['rows_different_native']+=int(native_diff>0);agg['different_native_elements']+=native_diff;agg['rows_different_stable_ideal']+=int(ideal_diff>0);agg['rank_inversions']+=int(not rank_valid);agg['nucleus_mass_below_p']+=int(not nucleus_valid);agg['nonminimal_count']+=int(not minimal_count);agg['max_mass_shortfall']=max(agg['max_mass_shortfall'],max(0.,pval-mass));agg['max_mask_difference_native']=max(agg['max_mask_difference_native'],native_diff)
proof=dict(complete=True,CPU_only=True,original_GPU_sources_and_results_unchanged=True,bitwise_native_qualification_not_changed=True,reference='CPU FP64 ideal softmax and stable descending nucleus, actual FP32 p; native approximations and nonfinite cases distinguished.',valid_finite_rows_across_all_labels=len(rows),summary=summary,rows=rows,raw_result_sha256=hashlib.sha256((a.raw/'result.json').read_bytes()).hexdigest(),analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
a.output.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(summary))
