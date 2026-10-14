# SPDX-License-Identifier: Apache-2.0
"""CPU fake-JIT proof that only the four intended TopP launch arguments change."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from router import PLANS, Router

calls=[]
class Kernel:
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            calls.append((tuple(grid),args,kwargs))
            return NS(metadata=NS(num_warps=kwargs['num_warps'],num_stages=kwargs['num_stages'],shared=0),n_regs=64,n_spills=0)
        return launch

rows=[]
for plan in PLANS:
    original=dict(VOCAB_SIZE=130560,BATCH_SIZE=64,TOPK_ENABLED=False,TOPP_ENABLED=True,
                  BLOCK_SIZE=8192,BLOCK_SIZE_TRUNC=4096,MASK_VALUE=-float('inf'))
    arguments=tuple(range(7))
    router=Router(Kernel(),plan)
    router[(64,)](*arguments,**original)
    grid,args,actual=calls[-1]
    assert grid==(64,) and args==arguments
    expected=dict(original,BLOCK_SIZE=plan[1],BLOCK_SIZE_TRUNC=plan[2],num_warps=plan[3],num_stages=1)
    assert actual==expected and original['BLOCK_SIZE']==8192
    assert router.metadata['num_warps']==plan[3]
    rows.append(dict(plan=plan,metadata=router.metadata))
base=Path(__file__).resolve().parent
output=dict(complete=True,cpu_only=True,gpu_execution=False,actual_Router_executed=True,
            fake_JIT_not_runtime_evidence=True,configurations=rows,
            unchanged_grid_positional_arguments_topk_topp_vocab_and_mask=True,
            sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),base/'router.py',base/'probe_tiles.py')})
out=base/'results/20261014/prepared_32g/CPU_ROUTER_AUDIT_32g.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k!='configurations'}))
