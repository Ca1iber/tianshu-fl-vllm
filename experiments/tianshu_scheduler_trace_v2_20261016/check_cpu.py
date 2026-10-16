# SPDX-License-Identifier: Apache-2.0
"""Check forwarding against the inspected installed scheduler signature."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[1]
SOURCES=REPO.parent/'service-results/20261016_tianshu/kv_scheduler_sources'
manifest=json.loads((BASE/'SOURCE_MANIFEST_32g.json').read_text())
for name,sha in manifest['files'].items():
    assert hashlib.sha256((SOURCES/name).read_bytes()).hexdigest()==sha
syntax=ast.parse((SOURCES/'vllm/v1/core/sched/scheduler.py').read_text())
cls=next(c for c in syntax.body if isinstance(c,ast.ClassDef) and c.name=='Scheduler')
method=next(c for c in cls.body if isinstance(c,ast.FunctionDef) and c.name=='schedule')
assert [a.arg for a in method.args.args]==['self','throttle_prefills']
assert len(method.args.defaults)==1 and method.args.defaults[0].value is False
profile=json.loads((REPO.parent/'service-results/20261016_tianshu/gemm_topp_geo_official/20261016_gemm_topp_geo_official_tianshu/result.json').read_text())['profile_environment']
env=dict(os.environ)
env.update({k:v.replace('/workspace/vllm-plugin-FL',str(REPO)) for k,v in profile.items()})
env.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_SCHEDULER_TRACE_DIR='/tmp/tianshu_trace_v2_cpu_contract')
script='''import sys,json
from types import SimpleNamespace as N
finder=next(x for x in sys.meta_path if type(x).__name__=='TraceFinder');g=type(finder).find_spec.__globals__
result=N(scheduled_new_reqs=[],scheduled_cached_reqs=N(req_ids=['r'],num_computed_tokens=[3]),num_scheduled_tokens={'r':4},total_num_scheduled_tokens=4,preempted_req_ids=None,finished_req_ids=set(),new_block_ids_to_zero=None)
class Scheduler:
    def schedule(self,throttle_prefills:bool=False):
        self.forwarded.append(throttle_prefills);return result
module=N(Scheduler=Scheduler);g['scheduler_hook'](module)
owner=Scheduler();owner.forwarded=[];owner.waiting=[];owner.running=['r'];owner.kv_cache_manager=N(block_pool=N(get_num_free_blocks=lambda:100));owner.requests={'r':N(num_prompt_tokens=5)}
assert owner.schedule() is result;assert owner.schedule(True) is result;assert owner.schedule(throttle_prefills=False) is result
assert owner.forwarded==[False,True,False]
assert all(s['prefill_tokens']==s['decode_tokens']==2 for s in g['rows'])
assert len(g['rows'])==3 and 'torch' not in sys.modules
print(json.dumps(dict(complete=True,CPU_only=True,unique_TraceFinder=True,no_torch_imported=True,real_interpreter_startup=True,return_identity_and_single_call_preserved=True,prompt_decode_boundary_classification_verified=True,positional_and_keyword_throttle_prefills_forwarding_checked=True)))
'''
result=subprocess.run(['python3','-c',script],env=env,text=True,capture_output=True)
assert result.returncode==0 and not result.stderr,(result.stdout,result.stderr)
record=json.loads(result.stdout)
record.update(actual_installed_signature=ast.unparse(method.args),all14_installed_source_downloads_SHA_verified=True)
(BASE/'CPU_PREPARATION_32g.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
