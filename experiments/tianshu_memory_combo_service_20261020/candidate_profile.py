# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_memory_combo_arena_profile',BASE.parent/'tianshu_small_arena_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'] and gate['CPU_only'];frozen_check(gate['sources'])
    seq=BASE.parent/'tianshu_seq_limit_default_20261020';seq_gate=read(seq/'CPU_GATE_32g.json');assert seq_gate['complete'];frozen_check(seq_gate['sources'])
    command,environment,profile,sources=parent.prepare(root,mode)
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_SEQ_LIMIT_MODE=mode,TIANSHU_SEQ_LIMIT_RECORD_DIR=str(root/'seq_limit_records'))
    assert len(profile)==50 and profile['TIANSHU_SMALL_ARENA_MODE']==mode and profile['TIANSHU_LONG_SEGMENT_MODE']=='candidate'
    assert profile['TIANSHU_MAX_SEQS_MODE']=='control' and 'TIANSHU_KV_HEADROOM_MODE' not in profile
    environment.update(profile)
    extra=[*BASE.rglob('*.py'),BASE/'CPU_GATE_32g.json',BASE/'batch_policy.json',*seq.glob('*.py'),seq/'CPU_GATE_32g.json',
           Path('/usr/local/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py'),
           Path('/usr/local/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py')]
    for path in extra:sources[str(path)]=digest(path)
    frozen_check(sources)
    for name,sha in sources.items():
        dest=root/'source_snapshot'/Path(name).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes());assert digest(dest)==sha
    return command,environment,profile,sources
