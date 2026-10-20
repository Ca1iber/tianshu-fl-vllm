# SPDX-License-Identifier: Apache-2.0
"""Actual native allocator API fallbacks, exceptions and whole recursive manager pickle."""
import dataclasses,hashlib,inspect,json,pickle
from pathlib import Path
import torch
from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher,init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec,KVCacheConfig,KVCacheGroupSpec
from vllm.v1.request import Request,RequestStatus
from shortcut import wrap
BASE=Path(__file__).resolve().parent

def hash_fn(x):return hashlib.sha256(pickle.dumps(x,protocol=5)).digest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert torch.cuda.device_count()==0 and not torch.cuda.is_initialized()
    init_none_hash(hash_fn);hasher=get_request_block_hasher(16,hash_fn);params=SamplingParams(max_tokens=128)
    def make():
        spec=FullAttentionSpec(block_size=16,num_kv_heads=2,head_size=128,dtype=torch.uint8,page_size_padded=12288)
        config=KVCacheConfig(num_blocks=128,kv_cache_tensors=[],kv_cache_groups=[KVCacheGroupSpec(layer_names=[f'layer{i}' for i in range(42)],kv_cache_spec=spec)])
        m=KVCacheManager(config,131072,16,16,max_num_batched_tokens=4096)
        r=Request('guard',list(range(64)),params,None,arrival_time=1.,block_hasher=hasher)
        assert m.allocate_slots(r,64) is not None;r.num_computed_tokens=65;r.status=RequestStatus.RUNNING;r.append_output_token_ids([42,43]);assert m.allocate_slots(r,1) is not None
        return m,r
    cases=[('default',{},None,1),('no_scheduled',dict(has_scheduled_reqs=False),None,1),('two_tokens',dict(num_new_tokens=2),None,0),
        ('invalid_zero',dict(num_new_tokens=0),None,0),('new_computed_token',dict(num_new_computed_tokens=1),None,0),
        ('new_computed_blocks',dict(new_computed_blocks='EMPTY'),None,0),('lookahead',dict(num_lookahead_tokens=1),None,0),
        ('external',dict(num_external_computed_tokens=1),None,0),('delay_cache',dict(delay_cache_blocks=True),None,0),
        ('encoder',dict(num_encoder_tokens=1),None,0),('admission',dict(full_sequence_must_fit=True),None,0),
        ('reservation',dict(reserved_blocks=1),None,0),('reservation_exhausted',dict(reserved_blocks=10000),None,0),
        ('waiting',{},'WAITING',0),('preempted',{},'PREEMPTED',0),('computed_limit',{},'LIMIT',0),
        ('partial_prefill',{},'PREFILL',0),('unknown_keyword',dict(unsupported_keyword=True),None,0),
        ('invalid_request',{},'NONE',0)]
    def norm(x):
        if x is None or isinstance(x,(str,int,float,bool,bytes)):return x
        if isinstance(x,dict):return tuple(sorted((repr(k),norm(v)) for k,v in x.items()))
        if isinstance(x,(list,tuple)):return tuple(norm(v) for v in x)
        if callable(x):return ('CALLABLE',getattr(x,'__module__',None),getattr(x,'__qualname__',None))
        if dataclasses.is_dataclass(x):return (type(x).__name__,tuple((f.name,norm(getattr(x,f.name))) for f in dataclasses.fields(x)))
        if hasattr(x,'__dict__'):return (type(x).__name__,norm(vars(x)))
        return repr(x)
    rows=[]
    for name,kwargs,change,hits in cases:
        pair=[make(),make()];m,r=pair[1]
        candidate,stats=wrap(m,m.allocate_slots,Request,RequestStatus.RUNNING,'candidate')
        assert inspect.signature(candidate)==inspect.signature(m.allocate_slots)
        functions=[pair[0][0].allocate_slots,candidate];answers=[]
        for (manager,req),fn in zip(pair,functions):
            if change in ('WAITING','PREEMPTED'):req.status=getattr(RequestStatus,change)
            if change=='LIMIT':req.num_computed_tokens=131072
            if change=='PREFILL':req.num_computed_tokens=63
            selected=None if change=='NONE' else req
            kw={k:(manager.empty_kv_cache_blocks if v=='EMPTY' else v) for k,v in kwargs.items()};tokens=kw.pop('num_new_tokens',1)
            try:
                value=fn(selected,tokens,**kw)
                answer=dict(returned=None if value is None else value.get_block_ids(),native_empty_singleton=value is manager.empty_kv_cache_blocks)
            except Exception as error:answer=dict(error_type=type(error).__name__,message=str(error))
            answers.append(answer)
        assert answers[0]==answers[1],(name,answers)
        assert pickle.dumps(pair[0][0],protocol=5)==pickle.dumps(pair[1][0],protocol=5),name
        assert norm(vars(pair[0][1]))==norm(vars(pair[1][1])),name
        assert stats['shortcuts']==hits,(name,stats)
        rows.append(dict(case=name,answer=answers[0],stats=stats,complete_recursive_manager_pickle_identical=True,full_Request_field_values_equal=True))
    assert torch.cuda.device_count()==0 and not torch.cuda.is_initialized()
    source_paths=[Path(__file__),BASE/'shortcut.py',Path(inspect.getsourcefile(KVCacheManager)),Path(inspect.getsourcefile(Request)),Path(inspect.getsourcefile(get_request_block_hasher))]
    gate=dict(complete=True,CPU_only=True,visible_GPUs=0,actual_native_classes_executed=True,api_signature_exact=True,
        guard_and_fallback_cases=rows,all_native_return_objects_values_and_exceptions_equal=True,
        whole_manager_recursive_pickle_and_Request_field_values_equal=True,no_GPU_context_or_tensor_or_kernel=True,
        sources={str(p.resolve()):sha(p) for p in source_paths})
    (BASE/'CPU_GUARD_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','guard_and_fallback_cases')}))
if __name__=='__main__':main()
