# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
EXP=BASE.parent
PARENT=EXP/'tianshu_r5_BM8_KV_20261024'
spec=importlib.util.spec_from_file_location('_BM8_capacity_parent_profile',PARENT/'candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check,merge=parent.digest,parent.read,parent.frozen_check,parent.merge
released_sources,BM8,PAIR_SHA=parent.released_sources,parent.BM8,parent.PAIR_SHA

def prepare(root,mode='candidate',check_gate=True):
    command,env,profile,sources=parent.prepare(root,mode)
    assert len(sources)==1023
    inherited=set(sources)
    paths=set(BASE.rglob('*.py'))|{BASE/'PLAN_32g.md',Path('/usr/local/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py')}
    merge(sources,{str(p):digest(p) for p in paths})
    if check_gate:
        gate=read(BASE/'CPU_GATE_32g.json')
        assert gate['complete'] and gate['model_ready'] and gate['sources']==sources
        sources[str(BASE/'CPU_GATE_32g.json')]=digest(BASE/'CPU_GATE_32g.json')
    frozen_check(sources)
    for p,h in sources.items():
        target=root/'source_snapshot'/Path(p).relative_to('/')
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes())
        assert (p in inherited or not target.is_symlink()) and digest(target)==h
    profile.update(PYTHONPATH=str(BASE/'hooks_candidate'),TIANSHU_KV_CAPACITY_QUANTUM='256',
                   TIANSHU_KV_CAPACITY_RECORD_DIR=str(root/'capacity_records'),VLLM_CACHE_ROOT=str(root/'vllm_cache'))
    env.update(profile)
    return command,env,profile,sources
