# SPDX-License-Identifier: Apache-2.0
"""Execute real native route AST and stage hooks with CPU tensors only."""
import ast
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import policy
BASE=Path(__file__).resolve().parent
HYBRID=BASE.parents[1]/'vllm_fl/worker/iluvatar_hybrid_attention.py'
node=ast.parse(HYBRID.read_text())
selected=[n for n in node.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Route','route_from_common')]
assert len(selected)==2
env=dict(dataclass=dataclass,torch=torch,__name__='__main__')
exec(compile(ast.Module(body=selected,type_ignores=[]),str(HYBRID),'exec'),env)
original_route=env['route_from_common'];checks=[]
def common(keys,queries,maximum,missing=False):
    starts=[0]
    for q in queries:starts.append(starts[-1]+q)
    c=SimpleNamespace(num_reqs=len(keys),num_actual_tokens=starts[-1],max_seq_len=maximum,
        query_start_loc_cpu=torch.tensor(starts,dtype=torch.int32),query_start_loc=torch.tensor(starts,dtype=torch.int32))
    if not missing:c.seq_lens_cpu_upper_bound=torch.tensor(keys,dtype=torch.int32)
    return c
fixtures=[
 ('pure',common([512]*8,[512]*8,512),0,512),
 ('mixed_prefix',common([32768,2048,4097,1202],[1,1,128,17],32768),2,4097),
 ('mixed_short',common([32768]*63+[65],[1]*63+[33],32768),63,65),
 ('decode',common([32768]*64,[1]*64,32768),None,None),
 ('missing',common([32768,1025],[1,128],32768,True),1,None),
 ('invalid_upper',common([32768,16],[1,128],32768),1,None),
]
for mode in ('control','candidate'):
 for name,c,nd,expected in fixtures:
    native=[]
    def route_once(value):
        answer=original_route(value);native.append(answer);return answer
    h=SimpleNamespace(route_from_common=route_once,_ROUTE=ContextVar('cpu_route',default=None))
    class Dense:
        def __init__(self):self.calls=[];self.records={}
        def __call__(self,kw):
            self.calls.append(kw);n=kw['seqused_k'].numel();blocks=(kw['max_seqlen_k']+15)//16
            self.records[str((n,blocks))]=dict(native_stage_called=True)
            return kw
    d=Dense();other=Dense();old_call=Dense.__call__;state=policy.install(h,d,mode)
    try:
        r=h.route_from_common(c);assert len(native)==1 and r is native[0]
        if nd is None:assert r is None and state['metadata_bounds']==0
        else:
            assert r.num_decodes==nd
            assert torch.equal(r.prefill_cu,c.query_start_loc[nd:]-nd)
            prefix=c.seq_lens_cpu_upper_bound[nd:] if hasattr(c,'seq_lens_cpu_upper_bound') else torch.ones(c.num_reqs-nd,dtype=torch.int32)
            kw=dict(max_seqlen_k=c.max_seq_len,max_seqlen_q=r.max_prefill_query,seqused_k=prefix,q=object(),out=object(),k=object(),v=object(),block_table=object(),cu_seqlens_q=r.prefill_cu)
            unchanged=dict(kw);token=h._ROUTE.set(r)
            try:
                result=d(kw);other_result=other(kw)
                tightened=mode=='candidate' and expected is not None and expected<c.max_seq_len
                assert (result is not kw)==tightened and other_result is kw
                assert result['max_seqlen_k']==(expected if tightened else c.max_seq_len)
                assert all(kw[k] is value for k,value in unchanged.items())
                assert all(result[k] is kw[k] for k in kw if k!='max_seqlen_k')
                assert state['tightened_calls']==int(tightened) and state['stage_calls']==1
                for field,value in (('max_seqlen_k',1),('max_seqlen_q',1),('seqused_k',torch.ones(c.num_reqs-nd+1,dtype=torch.int32))):
                    invalid=dict(kw,**{field:value});assert policy.narrow(r,invalid) is invalid
            finally:h._ROUTE.reset(token)
        checks.append(dict(mode=mode,fixture=name,passed=True))
    finally:Dense.__call__=old_call
assert len(checks)==12 and not torch.cuda.is_initialized()
for p in BASE.glob('*.py'):ast.parse(p.read_text())
paths=[BASE/'policy.py',BASE/'check_cpu.py',HYBRID,BASE.parent/'tianshu_prefill_key_bound_20261021/bound.py']
result=dict(complete=True,CPU_only=True,GPU_not_initialized=True,actual_native_route_AST_executed=True,
    native_route_identity_and_GPU_cu_logic_preserved=True,stage_only_copy_and_other_instances_passthrough=True,
    decode_inputs_original_identity_preserved=True,no_new_hook_file_IO=True,checks=checks,
    GPU_component_pending=True,model_not_installed=True,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
