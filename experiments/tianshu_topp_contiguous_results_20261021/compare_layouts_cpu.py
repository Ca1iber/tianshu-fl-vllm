# SPDX-License-Identifier: Apache-2.0
"""Prove whether previous mathematical diagnostics apply to identical full outputs."""
import hashlib,json
from pathlib import Path
import torch
torch.set_num_threads(8)
OLD=Path('/workspace/benchmark-results/20261021_topp_launch_GPU_tianshu');NEW=Path('/workspace/benchmark-results/20261021_topp_contiguous_GPU_tianshu')
old=json.loads((OLD/'result.json').read_text());new=json.loads((NEW/'result.json').read_text());assert old['complete'] and new['complete'] and len(old['cases'])==len(new['cases'])==12
same=True;compared=0;phases=0
for a,b in zip(old['cases'],new['cases'],strict=True):
 assert a['name']==b['name'] and b['actual_logit_row_stride']==130560
 for x,y in zip(a['phases'],b['phases'],strict=True):
  av=torch.load(OLD/x['archive'],map_location='cpu',weights_only=True);bv=torch.load(NEW/y['archive'],map_location='cpu',weights_only=True)
  assert torch.equal(av['input_bf16'].view(torch.int16),bv['input_bf16'].view(torch.int16)) and torch.equal(av['p'].view(torch.int32),bv['p'].view(torch.int32));phases+=1
  for label,value in av['outputs'].items():
   other=bv['outputs'][label];equal=value['full_output_sha256']==other['full_output_sha256']
   equal=equal and torch.equal(value['keep_mask_packbits_little'],other['keep_mask_packbits_little']);same=same and equal;compared+=1
result=dict(complete=True,CPU_only=True,old_raw_root=str(OLD),new_raw_root=str(NEW),old_raw_result_sha256=hashlib.sha256((OLD/'result.json').read_bytes()).hexdigest(),new_raw_result_sha256=hashlib.sha256((NEW/'result.json').read_bytes()).hexdigest(),same_original_input_and_p_phases=phases,full_output_pairs=compared,all_full_output_SHA_and_masks_identical=same,old_mathematical_diagnostics_apply_to_these_identical_outputs=same,old_bitwise_native_failure_not_reclassified=True,does_not_prove_new_model_accuracy_or_all_unseen_inputs=True,analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out=Path(__file__).resolve().parent/'verified/LAYOUT_NUMERICAL_COMPARISON.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
