# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent;meta=Path(torch.__file__).parent/'_meta_registrations.py';rows=[]
for name in sorted(n for n in torch._C._dispatch_get_all_op_names() if n.startswith('aten::') and ('int4' in n or '_weight_int8pack_mm' in n)):
 row={'name':name}
 try:row.update(CUDA=torch._C._dispatch_has_kernel_for_dispatch_key(name,'CUDA'),CPU=torch._C._dispatch_has_kernel_for_dispatch_key(name,'CPU'),registration=torch._C._dispatch_dump(name))
 except Exception as error:row['error']=repr(error)
 rows.append(row)
assert not torch.cuda.is_initialized()
result=dict(complete=True,CPU_only=True,CUDA_initialized=False,torch_version=torch.__version__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),meta_source=str(meta),meta_sha256=hashlib.sha256(meta.read_bytes()).hexdigest(),operators=rows,GPU_functionality_not_tested=True,performance_not_tested=True,not_a_model_candidate=True)
(BASE/'CPU_REGISTRY_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='operators'}));print(json.dumps([{k:v for k,v in row.items() if k!='registration'} for row in rows]))
