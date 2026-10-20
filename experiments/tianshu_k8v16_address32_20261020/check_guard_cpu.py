# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import guard
BASE=Path(__file__).resolve().parent
root=Path('/workspace/benchmark-results/20261020_k8v16_address32_guard_CPU_tianshu');root.mkdir(exist_ok=False)
class Kernel:
    def __init__(self,name):self.name=name;self.calls=[]
    def __getitem__(self,grid):
        def run(**kwargs):self.calls.append((grid,kwargs));return self.name
        return run
small=SimpleNamespace(shape=(43166,16,2,128),stride=lambda:(12288,256,128,1))
huge=SimpleNamespace(shape=(2**31,),stride=lambda:(1,))
scalar=SimpleNamespace(shape=(),stride=lambda:())
rows=[]
for name,change,expected in [
    ('selected',{},'narrow'),('wide_cache',dict(key_cache_ptr=huge),'wide'),
    ('scalar_descale',dict(k_scale=scalar),'narrow'),('2D',dict(IS_3D=False),'wide'),
    ('batch65',dict(num_seqs=65),'wide'),('other_quant',dict(KV_QUANT_MODE=1),'wide'),
    ('TD',dict(USE_TD=True),'wide'),('sinks',dict(USE_SINKS=True),'wide'),
    ('wide_integer_stride',dict(stride_k_cache_0=2**31),'wide')]:
    a,b=Kernel('wide'),Kernel('narrow');selector=guard.SelectingKernel(a,b,'wide_sha','narrow_sha')
    kwargs=dict(IS_3D=True,num_seqs=64,KV_QUANT_MODE=0,key_cache_ptr=small);kwargs.update(change)
    answer=selector[(64,2,16)](**kwargs);assert answer==expected
    selected=a if expected=='wide' else b
    assert len(a.calls)+len(b.calls)==1 and selected.calls[0]==((64,2,16),kwargs)
    assert selector.last_selection['address_bits']==(64 if expected=='wide' else 32)
    rows.append(dict(case=name,**selector.last_selection))
sources=[Path(__file__),BASE/'guard.py',BASE/'source.py']
result=dict(complete=True,CPU_only=True,actual_selector_executed=True,cases=rows,
            unsafe_inputs_use_original_wide_kernel=True,arguments_return_values_and_grid_forwarded=True,
            no_torch_or_GPU_imported=True,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(root/'GUARD_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');(BASE/'GUARD_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
for p in sources:
    dst=root/'source_snapshot'/p.resolve().relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
print(json.dumps(dict(complete=True,CPU_only=True,cases=len(rows))))
