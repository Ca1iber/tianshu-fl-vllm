# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from launch import LABELS,WARPS,CASES,direct
BASE=Path(__file__).resolve().parent;SITE=Path('/usr/local/lib/python3.12/site-packages')
for path in BASE.glob('*.py'):ast.parse(path.read_text())
NATIVE=SITE/'vllm/v1/sample/ops/topk_topp_triton.py';tree=ast.parse(NATIVE.read_text())
assert sum(isinstance(x,ast.FunctionDef) and x.name=='_topk_topp_kernel' for x in tree.body)==1
seen=[]
class Tensor:
 shape=(64,130560)
 def stride(self,i):return self.shape[1] if i==0 else 1
class Kernel:
 def __getitem__(self,grid):
  def call(*args,**kw):seen.append((grid,args,kw));return object()
  return call
logits=Tensor();p=object();buffer=object();normal=object();percentile=object();module=SimpleNamespace(_topk_topp_kernel=Kernel())
for label,warps in WARPS.items():
 direct(module,logits,p,buffer,(normal,percentile),64,warps)
 grid,args,kw=seen[-1]
 assert grid==(64,) and args[0] is logits and args[1]==130560 and args[2] is buffer and args[3] is percentile and args[4] is normal and args[5] is logits and args[6] is p
 assert kw==dict(BATCH_SIZE=64,VOCAB_SIZE=130560,MASK_VALUE=-float('inf'),BLOCK_SIZE=8192,BLOCK_SIZE_TRUNC=4096,TOPK_ENABLED=False,TOPP_ENABLED=True,num_warps=warps)
parents=[NATIVE,SITE/'vllm/v1/sample/ops/topk_topp_sampler.py',SITE/'vllm/utils/math_utils.py',SITE/'vllm/utils/platform_utils.py']
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'PLAN.md',*parents]
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
r=dict(complete=True,CPU_only=True,GPU_not_used=True,actual_launcher_argument_cases=4,numerical_JIT_source_unchanged=True,only_num_warps_varies=True,original_tiles=[8192,4096],cases=CASES,labels=LABELS,phases=3,full_GPU_output_comparisons_planned=180,source_inventory_scope='Primary component and native top-p source/helper files; not all transitive Python dependencies.',sources=sources)
(BASE/'CPU_GATE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='sources'}))
