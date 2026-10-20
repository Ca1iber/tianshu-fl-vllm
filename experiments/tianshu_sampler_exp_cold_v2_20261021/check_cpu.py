# SPDX-License-Identifier: Apache-2.0
"""Exercise actual LibEntry key function across batch and signed-int64 ranges."""
import ast,hashlib,importlib,inspect,json
from pathlib import Path
import torch,flag_gems
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    for p in BASE.glob('*.py'):ast.parse(p.read_text())
    torch.set_num_threads(8)
    native=importlib.import_module('flag_gems.ops.exponential_');entry=native.fused_exponential_kernel_f32_unroll8
    assert flag_gems.vendor_name=='iluvatar' and flag_gems.exponential_ is native.exponential_
    assert entry.arg_names==['out_ptr','N','inv_lambd','eps_minus','philox_seed','philox_offset','BLOCK']
    tensor=torch.empty(1,dtype=torch.float32);assert tensor.data_ptr()%16==0
    seeds=(-2**63,-2**31-1,-1,0,2**31-1,2**31,2**63-1)
    offsets=(-2**63,-2**31-4,-4,0,2**31-4,2**31,2**32,2**63-4)
    keys=set();count=0;observed={}
    for batch in range(1,65):
        numel=batch*130560;assert numel<2**31
        for seed in seeds:
            for offset in offsets:
                key=entry.key([tensor,1.,-.5*torch.finfo(torch.float32).eps],[numel,seed,offset],[])
                pair=tuple(key[-2:]);assert key[-3]=='i32' and pair[0] in ('i32','i64') and pair[1] in ('i32','i64')
                if pair in observed:assert observed[pair]==key
                else:observed[pair]=key
                keys.add(key);count+=1
    assert len(keys)==4 and count==3584
    paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),Path(native.__file__),Path(inspect.getsourcefile(type(entry))),Path('/usr/local/lib/python3.12/site-packages/flag_gems/utils/random_utils.py')]
    gate=dict(complete=True,CPU_only=True,no_GPU_work_launched=True,adopted=False,
        actual_native_LibEntry_key_executions=count,batches_checked=64,signed_seed_and_offset_range_boundaries_checked=True,
        actual_dispatch_classes=4,N_values_share_keys_within_i32_range=True,
        source_of_late_miss='Philox scalar type transition at 2^31, N37 belongs to first dispatch after transition',
        sources={str(p):sha(p) for p in paths})
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
if __name__=='__main__':main()
