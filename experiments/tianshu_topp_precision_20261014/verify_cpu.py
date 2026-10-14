# SPDX-License-Identifier: Apache-2.0
"""Actual routing and reversible source checks without GPU imports."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from router import PLANS, Router
from precision import render_source, topk_ast

calls=[]
class Kernel:
    def __init__(self,label):self.label=label;self.fn=NS(__module__=label)
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            calls.append((self.label,tuple(grid),args,kwargs))
            return NS(metadata=NS(num_warps=kwargs['num_warps'],num_stages=kwargs['num_stages'],shared=0),n_regs=64,n_spills=0)
        return launch

rows=[]
for topk in (False,True):
    for plan in PLANS:
        before=dict(VOCAB_SIZE=130560,BATCH_SIZE=64,TOPK_ENABLED=topk,TOPP_ENABLED=True,
                    BLOCK_SIZE=8192,BLOCK_SIZE_TRUNC=4096,MASK_VALUE=-float('inf'))
        router=Router(Kernel('native'),plan,corrected_kernel=Kernel('fixed'))
        router[(64,)](*range(7),**before)
        kind,grid,args,actual=calls[-1]
        corrected=plan[0].startswith('fixed') and not topk
        assert kind==('fixed' if corrected else 'native') and grid==(64,) and args==tuple(range(7))
        label,tile,trunc,warps=plan
        if topk and label.startswith('fixed'):tile,trunc,warps=8192,4096,16
        assert actual==dict(before,BLOCK_SIZE=tile,BLOCK_SIZE_TRUNC=trunc,num_warps=warps,num_stages=1)
        assert router.metadata['corrected_source_active']==corrected
        rows.append(dict(topk=topk,plan=plan,metadata=router.metadata))
base=Path(__file__).resolve().parent
raw=(base.parents[2]/'service-results/20261013_tianshu/topp_launch_cpu_reference/20261013_topp_launch_cpu_reference_tianshu/source_snapshot/native_topk_topp_triton.py').read_bytes()
source=render_source(raw)
assert topk_ast(source)==topk_ast(raw.decode())
output=dict(complete=True,cpu_only=True,gpu_execution=False,actual_router12_cases_verified=True,
            source_reversibility_and_TopK_AST_verified=True,generated_sha256=hashlib.sha256(source.encode()).hexdigest(),
            fake_JIT_not_runtime_evidence=True,configurations=rows,
            sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),base/'router.py',base/'precision.py',base/'probe_precision.py')})
out=base/'results/20261014/prepared_32g/CPU_AUDIT_32g.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k!='configurations'}))
