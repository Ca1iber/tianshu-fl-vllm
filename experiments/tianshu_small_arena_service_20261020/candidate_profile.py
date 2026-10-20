# SPDX-License-Identifier: Apache-2.0
import importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_arena_long_profile',BASE.parent/'tianshu_long_segments_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'] and gate['CPU_only'];frozen_check(gate['sources'])
    component=BASE.parent/'tianshu_small_arena_20261020'
    gpu=read(component/'GPU_GATE_32g.json');assert gpu['complete'] and gpu['independently_rechecked'];frozen_check(gpu['frozen_sources'])
    command,environment,profile,sources=parent.prepare(root,'candidate')
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_SMALL_ARENA_MODE=mode,TIANSHU_SMALL_ARENA_RECORD_DIR=str(root/'small_arena_records'))
    assert len(profile)==48 and profile['TIANSHU_LONG_SEGMENT_MODE']=='candidate' and 'TIANSHU_KV_HEADROOM_MODE' not in profile
    environment.update(profile)
    extra=[*BASE.rglob('*.py'),BASE/'CPU_GATE_32g.json',BASE/'batch_policy.json',
           *component.glob('*.py'),component/'CPU_GATE_32g.json',component/'GPU_GATE_32g.json']
    for path in extra:sources[str(path)]=digest(path)
    frozen_check(sources)
    for name,sha in sources.items():
        dest=root/'source_snapshot'/Path(name).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes());assert digest(dest)==sha
    return command,environment,profile,sources
