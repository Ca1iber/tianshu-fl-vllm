# SPDX-License-Identifier: Apache-2.0
"""CPU syntax/layout algebra; no Torch/Triton/FlagGems imports or GPU calls."""
import ast,hashlib,json
from pathlib import Path
base=Path(__file__).resolve().parent
sources=[base/name for name in ('__init__.py','packed_bundle.py','packed_swiglu.py','matched.py','probe_bf16.py','verify_bf16_cpu.py','check_bf16_cpu.py','run_bf16.sh','PLAN_32g.md','SOURCE_PREPARATION_32g.json')]
sources += [Path('/usr/local/lib/python3.12/site-packages/flag_gems/ops/mm.py'),Path('/usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar/ops/mm.py')]
for source in sources:
    if source.suffix=='.py':compile(source.read_text(),str(source),'exec')
assert (base/'matched.py').read_bytes()==Path('/workspace/FlagGems/experiments/tianshu_packing/matched_gemm.py').read_bytes()
module=ast.parse((base/'probe_bf16.py').read_text());assigns={node.targets[0].id:ast.literal_eval(node.value) for node in module.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('ROWS','ROLES')}
count=0
for m in assigns['ROWS']:
    for role,(n,k) in assigns['ROLES'].items():
        bn,bk=(64,64) if role=='down' else (128,32)
        assert n%bn==0 and k%bk==0
        for row in sorted({0,m//2,m-1}):
            for col in (0,1,bk-1,bk,k//2,k-1):
                physical=(col//bk)*m*bk+row*bk+col%bk
                group,inner=divmod(physical,m*bk);rr,cc=divmod(inner,bk)
                assert (rr,group*bk+cc)==(row,col);count+=1
        for row in (0,1,bn-1,bn,n//2,n-1):
            for col in (0,1,bk-1,bk,k//2,k-1):
                physical=((row//bn)*(k//bk)+col//bk)*bn*bk+(row%bn)*bk+col%bk
                tile,inner=divmod(physical,bn*bk);outer_n,outer_k=divmod(tile,k//bk);rr,cc=divmod(inner,bk)
                assert (outer_n*bn+rr,outer_k*bk+cc)==(row,col);count+=1
result=dict(complete=True,CPU_only=True,syntax_compiled_without_import=True,layout_index_roundtrips=count,
    roles=assigns['ROLES'],M_grid=assigns['ROWS'],same_matched_kernel_bytes_as_existing_Iluvatar_control=True,
    GPU_correctness_proven=False,service_executed=False,scope='BF16 QKV/O/Down isolated stages; INT8 GateUp/LM and fused MLP drivers still separate preparation',
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
(base/'CPU_BF16_PREPARATION_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key!='sources'},indent=2))
