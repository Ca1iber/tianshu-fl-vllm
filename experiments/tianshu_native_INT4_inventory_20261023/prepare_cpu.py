# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
assert not (BASE/'CPU_PREPARE_32g.json').exists() and not torch.cuda.is_initialized()
for p in BASE.glob('*.py'):ast.parse(p.read_text())
x=torch.empty((64,256),device='meta',dtype=torch.bfloat16);w=torch.empty((256,128),device='meta',dtype=torch.uint8);params=torch.empty((2,256,2),device='meta',dtype=torch.bfloat16);packed=torch.ops.aten._convert_weight_to_int4pack.default(w,8);out=torch.ops.aten._weight_int4pack_mm.default(x,packed,128,params);assert list(packed.shape)==[32,2,32,4] and list(out.shape)==[64,256] and out.dtype==torch.bfloat16
paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run_smoke.sh',BASE/'CPU_REGISTRY_32g.json',Path(torch.__file__).parent/'_meta_registrations.py'};sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};assert not torch.cuda.is_initialized();result=dict(complete=True,CPU_only=True,CUDA_initialized=False,torch_version=torch.__version__,actual_Meta_shapes_passed=True,input_shape=[64,256],unsigned_packed_input_shape=[256,128],converted_packed_shape=[32,2,32,4],group=128,inner_k_tiles=8,params_shape=[2,256,2],nibbles=[7,8,9,10],predeclared_output_scalars=[-4,0,4,8],expected_total_output_cells=65536,GPU_not_run=True,performance_not_tested=True,sources=sources);(BASE/'CPU_PREPARE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
