# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_headroom_long_profile',BASE.parent/'tianshu_long_segments_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'] and gate['CPU_only']
    frozen_check(gate['sources'])
    command,environment,profile,sources=parent.prepare(root,'candidate')
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_KV_HEADROOM_MODE=mode,
                   TIANSHU_KV_HEADROOM_RECORD_DIR=str(root/'kv_headroom_records'))
    assert len(profile)==48 and profile['TIANSHU_LONG_SEGMENT_MODE']=='candidate'
    environment.update(profile)
    extra=[*BASE.rglob('*.py'),BASE/'CPU_GATE_32g.json',BASE/'batch_policy.json',
           BASE.parent/'tianshu_kv_headroom_20261020/policy.py',
           BASE.parent/'tianshu_kv_headroom_20261020/CPU_GATE_32g.json',
           Path('/usr/local/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py'),
           Path('/usr/local/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py')]
    for path in extra:sources[str(path)]=digest(path)
    frozen_check(sources)
    for name,sha in sources.items():
        destination=root/'source_snapshot'/Path(name).relative_to('/')
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(Path(name).read_bytes())
        assert digest(destination)==sha
    return command,environment,profile,sources
