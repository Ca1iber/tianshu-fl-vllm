# SPDX-License-Identifier: Apache-2.0
"""Metadata contract and actual native dense-page arithmetic, CPU only."""
import ast,dataclasses,hashlib,json,random
from pathlib import Path
from types import SimpleNamespace
import torch
from bound import derive
BASE=Path(__file__).resolve().parent
NATIVE=BASE.parent/'tianshu_k8v16_pipeline_20261019/dense_prefill.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not torch.cuda.is_initialized();torch.set_num_threads(4)
node=next(n for n in ast.parse(NATIVE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='shape_for')
namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(NATIVE),'exec'),namespace);shape_for=namespace['shape_for']
def make(queries,keys,decodes,maximum=None):
    cumulative=[0]
    for q in queries:cumulative.append(cumulative[-1]+q)
    common=SimpleNamespace(num_reqs=len(queries),num_actual_tokens=cumulative[-1],max_seq_len=maximum if maximum is not None else max(keys),query_start_loc_cpu=torch.tensor(cumulative,dtype=torch.int32),seq_lens_cpu_upper_bound=torch.tensor(keys,dtype=torch.int32))
    route=SimpleNamespace(num_decodes=decodes,max_prefill_query=max(queries[decodes:]))
    return common,route
cases=[]
for queries,keys,nd in (([1,128],[32768,128],1),([1,64,128],[16384,4096,8192],1),([128,256],[4096,1024],0),([1,1,2,17],[8192,32768,16,129],2)):
    common,route=make(queries,keys,nd);value=derive(common,route);assert value and value.max_key_length==max(keys[nd:]) and value.num_prefills==len(queries)-nd
    cases.append(dataclasses.asdict(value))
common,route=make([1,64,128],[16384,4096,8192],1)
invalid=[]
for name,change in (
 ('missing_bounds',lambda c,r:setattr(c,'seq_lens_cpu_upper_bound',None)),('missing_starts',lambda c,r:setattr(c,'query_start_loc_cpu',None)),
 ('floating_bounds',lambda c,r:setattr(c,'seq_lens_cpu_upper_bound',c.seq_lens_cpu_upper_bound.float())),
 ('short_bounds',lambda c,r:setattr(c,'seq_lens_cpu_upper_bound',c.seq_lens_cpu_upper_bound[:1])),
 ('short_starts',lambda c,r:setattr(c,'query_start_loc_cpu',c.query_start_loc_cpu[:2])),
 ('bad_start',lambda c,r:c.query_start_loc_cpu.__setitem__(0,1)),('bad_total',lambda c,r:setattr(c,'num_actual_tokens',1)),
 ('bad_decode_prefix',lambda c,r:setattr(r,'num_decodes',2)),('too_short_key',lambda c,r:c.seq_lens_cpu_upper_bound.__setitem__(1,1)),
 ('beyond_global_bound',lambda c,r:c.seq_lens_cpu_upper_bound.__setitem__(1,32769)),('wrong_max_query',lambda c,r:setattr(r,'max_prefill_query',129))):
    c,r=make([1,64,128],[16384,4096,8192],1);change(c,r);assert derive(c,r) is None,name;invalid.append(name)
class NeverRead:
    def tolist(self):raise AssertionError('unsupported metadata read')
    def to(self,*a,**k):raise AssertionError('unexpected transfer')
c,r=make([1,64],[32768,128],1);c.seq_lens_cpu_upper_bound=NeverRead();assert derive(c,r) is None
assert derive(common,None) is None
# Native padded rows are ignored only when outside common.num_reqs. A padded
# zero query inside the claimed active rows remains invalid.
c,r=make([1,64],[32768,128],1);c.seq_lens_cpu_upper_bound=torch.cat((c.seq_lens_cpu_upper_bound,torch.zeros(3,dtype=torch.int32)));assert derive(c,r).max_key_length==128
c,r=make([1,64,0],[32768,128,0],1);assert derive(c,r) is None
rng=random.Random(20261021);generated=0
for _ in range(2048):
    nd=rng.randrange(0,57);np=rng.randrange(1,min(8,64-nd)+1)
    queries=[1]*nd+[rng.randrange(2,8193) for _ in range(np)]
    keys=[rng.randrange(1,131073) for _ in range(nd)]+[rng.randrange(q,131073) for q in queries[nd:]]
    c,r=make(queries,keys,nd);result=derive(c,r);assert result and result.max_key_length==max(keys[nd:])
    old_blocks,old_pages=shape_for(np,c.max_seq_len);new_blocks,new_pages=shape_for(np,result.max_key_length)
    assert new_pages<=old_pages and all((k+15)//16<=new_blocks for k in keys[nd:]);generated+=1
checked_pages=0
for np in range(1,9):
    for length in (2,15,16,17,4096,16384,32768):
        keys=[max(2,length-i) for i in range(np)];old_blocks,_=shape_for(np,131072);new_blocks,_=shape_for(np,max(keys))
        old=set();new=set()
        for seq,k in enumerate(keys):
            for page in range((k+15)//16):
                a=seq*old_blocks+page;b=seq*new_blocks+page
                assert a not in old and b not in new;old.add(a);new.add(b);checked_pages+=1
        assert len(old)==len(new) and len(new)<=np*new_blocks
assert not torch.cuda.is_initialized()
paths=[*BASE.glob('*.py'),NATIVE,BASE.parent/'vllm_fl/worker/iluvatar_hybrid_attention.py',Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/backend.py'),BASE.parent/'vllm_fl/worker/model_runner.py']
paths=[p if p.exists() else BASE.parents[1]/p.relative_to(BASE.parent) for p in paths]
proof=dict(complete=True,CPU_only=True,no_GPU_started=True,model_not_installed=True,GPU_gather_equivalence_not_yet_tested=True,model_Math_not_run=True,
    actual_native_shape_for_AST_executed=True,deterministic_generated_metadata_cases=generated,all_active_CPU_logical_page_addresses_checked=checked_pages,
    explicit_cases=cases,invalid_metadata_cases=invalid,unsupported_values_rejected_without_transfer=True,full_prefix_key_not_query_length_used=True,
    padding_outside_active_rows_ignored=True,padding_inside_active_rows_rejected=True,sources={str(p.resolve()):sha(p) for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='sources'}))
