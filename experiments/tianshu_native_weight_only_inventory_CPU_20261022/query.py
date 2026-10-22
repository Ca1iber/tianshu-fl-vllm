# SPDX-License-Identifier: Apache-2.0
"""CPU dispatcher/schema inventory, not hardware capability verification."""
import hashlib,inspect,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
assert not torch.cuda.is_initialized() and not (BASE/'CPU_INVENTORY_32g.json').exists()
rows=[]
for name in sorted(torch._C._dispatch_get_all_op_names()):
 if name.endswith('.out'):continue
 if (any(k in name for k in ('mm','gemm','linear')) and any(k in name for k in ('int8','int4','int_mm','scaled_mm','weight_only'))) or 'convert_weight_to_int' in name:
  try:cuda=torch._C._dispatch_has_kernel_for_dispatch_key(name,'CUDA')
  except RuntimeError:cuda=None
  rows.append(dict(operator=name,CUDA_dispatch_registered=cuda,CUDA_registration=[line for line in torch._C._dispatch_dump_table(name).splitlines() if line.startswith('CUDA:')],schemas=[str(schema) for schema in torch._C._jit_get_schemas_for_operator(name.split('.')[0])]))
meta=Path(torch.__file__).resolve().parent/'_meta_registrations.py'
with meta.open('rb') as stream:meta_sha=hashlib.file_digest(stream,'sha256').hexdigest()
assert not torch.cuda.is_initialized()
result=dict(complete=True,CPU_only=True,CUDA_uninitialized=True,hardware_execution_not_tested=True,torch_version=torch.__version__,Torch_path=torch.__file__,meta_source=str(meta),meta_source_sha256=meta_sha,operators=rows,model_read=False,adopted=False)
(BASE/'CPU_INVENTORY_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
