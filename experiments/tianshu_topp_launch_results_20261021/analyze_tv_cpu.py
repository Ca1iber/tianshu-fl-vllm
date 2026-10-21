# SPDX-License-Identifier: Apache-2.0
"""Exact CPU TV of two truncated distributions, preserving original mask failures."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
import numpy as np
import torch
torch.set_num_threads(8);r=json.loads((a.raw/'result.json').read_text());assert r['complete'];samples={k:[] for k in ('warp2','warp4','warp8','warp16')};diffs={k:0 for k in samples};rows=0
for case in r['cases']:
 for phase in case['phases']:
  data=torch.load(a.raw/phase['archive'],map_location='cpu',weights_only=True);x=data['input_bf16'].float();batch,vocab=x.shape
  masks={label:torch.from_numpy(np.unpackbits(out['keep_mask_packbits_little'].numpy(),bitorder='little',count=batch*vocab).copy()).bool().reshape(batch,vocab) for label,out in data['outputs'].items()}
  for i in range(batch):
   source=x[i].double()
   if abs(float(data['p'][i])-.95)>1e-7 or bool(torch.isnan(source).any() or torch.isposinf(source).any()) or not bool(torch.isfinite(source).any()):continue
   probs=source.softmax(0);native=masks['native'][i];native_mass=float(probs[native].sum());assert native_mass>0;rows+=1
   for label in samples:
    candidate=masks[label][i];candidate_mass=float(probs[candidate].sum());assert candidate_mass>0
    intersection=float(probs[native&candidate].sum());tv=max(0.,1.-intersection/max(native_mass,candidate_mass));samples[label].append(tv);diffs[label]+=int(not torch.equal(native,candidate))
summary={label:dict(rows=len(values),different_mask_rows=diffs[label],max_TV=max(values),mean_TV=sum(values)/len(values),p99_TV=sorted(values)[int(.99*(len(values)-1))]) for label,values in samples.items()}
proof=dict(complete=True,CPU_only=True,raw_result_sha256=hashlib.sha256((a.raw/'result.json').read_bytes()).hexdigest(),analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference='CPU FP64 original softmax, TV=1-overlap_mass/max(native_mass,candidate_mass); valid finite rows at actual FP32 p≈.95 only.',rows=rows,bitwise_native_failures_not_reclassified=True,does_not_prove_unseen_inputs_or_model_accuracy=True,summary=summary)
a.output.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
