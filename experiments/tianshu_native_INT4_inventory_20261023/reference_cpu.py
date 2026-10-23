# SPDX-License-Identifier: Apache-2.0
"""Independent native CPU semantics for the already frozen CUDA smoke inputs."""
import hashlib,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
assert not (BASE/'CPU_SEMANTICS_32g.json').exists() and not torch.cuda.is_initialized()
rows=[]
try:
 x=torch.full((64,256),.125,dtype=torch.bfloat16);params=torch.zeros((2,256,2),dtype=torch.bfloat16);params[:,:,0]=.125
 for nibble in (7,8,9,10):
  unpacked=torch.full((256,256),nibble,dtype=torch.int32);packed=torch.ops.aten._convert_weight_to_int4pack_for_cpu.default(unpacked,8);out=torch.ops.aten._weight_int4pack_mm_for_cpu.default(x,packed,128,params);expected=256*.125*(nibble-8)*.125;rows.append(dict(nibble=nibble,packed_shape=list(packed.shape),out_shape=list(out.shape),expected_scalar=expected,observed_min=float(out.min()),observed_max=float(out.max()),all_cells_exact=bool((out==expected).all()),cells=out.numel()))
 result=dict(complete=True,CPU_only=True,native_CPU_operations_supported=True,signed_center8_formula_confirmed=all(r['all_cells_exact'] for r in rows),rows=rows,CUDA_hardware_not_tested=True,performance_not_tested=True)
except Exception as error:result=dict(complete=True,CPU_only=True,native_CPU_operations_supported=False,error_type=type(error).__name__,error=str(error),rows=rows,CUDA_hardware_not_tested=True,performance_not_tested=True)
assert not torch.cuda.is_initialized();result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(BASE/'CPU_SEMANTICS_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
