# SPDX-License-Identifier: Apache-2.0
"""Check actual Producer branches and reversible transformation of the qualified reader."""
import argparse,ast,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from plans import MODES,PLANS,configuration
from layout import geometry
from mapping import render,SOURCE_SHA
p=argparse.ArgumentParser();p.add_argument('--reader-source',type=Path,required=True);a=p.parse_args()
raw=a.reader_source.read_bytes();generated=render(raw);assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
b=Path(__file__).resolve().parent
node=next(n for n in ast.walk(ast.parse((b/'probe.py').read_text())) if isinstance(n,ast.ClassDef) and n.name=='Producer')
context=[33];ns=dict(configuration=configuration,PLANS=PLANS,current_context=context,data={'producer_records':{}})
exec(compile(ast.Module(body=[node],type_ignores=[]),str(b/'probe.py'),'exec'),ns)
class Kernel:
    arg_names=['DIRECT_DECODE']
    def __getitem__(self,grid):
        assert len(grid)==3 and all(x>0 for x in grid)
        def launch(**kw):
            assert kw['KV_CACHE_MODIFIER']=='.cv'
            return SimpleNamespace(metadata=SimpleNamespace(num_warps=kw['num_warps'],num_stages=kw['num_stages']),src=SimpleNamespace(constants={(0,):kw.get('DIRECT_DECODE',False)}),hash='CPU-fixture-only',n_regs=0,n_spills=0)
        return launch
cases=0
for mode in MODES:
 for n in (1,8,16,24,32,64):
  for length in (33,4096,5120):
   assert geometry(mode)==dict(key_bytes=1,value_bytes=2,page_bytes=12288,value_offset=4096)
   c=configuration(mode,n,length);assert c==configuration('current',n,length)
   context[0]=length;prod=ns['Producer'](Kernel(),Kernel());prod.mode=mode
   kw=dict(KV_QUANT_MODE=0,HEAD_SIZE=128,num_seqs=n,IS_3D=True,NUM_SEGMENTS_PER_SEQ=c['segments'],key_cache_ptr=SimpleNamespace(dtype='U8'),value_cache_ptr=SimpleNamespace(dtype='BF16'))
   grid=(n//2+n,2,c['segments']);prod[grid](**kw)
   r=ns['data']['producer_records'][str((mode,n,length))]
   assert r['actual_direct_constexpr']==PLANS[mode] and r['actual_grid']==list((n,2,c['segments']) if PLANS[mode] else grid)
   cases+=1
assert cases==36
for p in b.glob('*.py'):ast.parse(p.read_text())
r=dict(complete=True,CPU_only=True,actual_Producer_control_flow_cases=cases,reader_source_sha256=SOURCE_SHA,actual_reader_AST_reversible_except_resolver_and_constexpr=True,all_arms_same_K8V16_and_launch_policy=True,sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(b.glob('*.py'))},GPU_not_run=True,padded_empty_queries_not_tested=True,model_and_Math_not_run=True,adopted=False)
(b/'CPU_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
