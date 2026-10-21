# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_memory_observation_static420',BASE.parent/'tianshu_static_compile_20261021/candidate_profile.py')
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
read,digest,frozen_check=parent.read,parent.digest,parent.frozen_check

def prepare(root,mode):
    command,environment,profile,sources=parent.prepare(root,mode);assert len(sources)==420 and len(profile)==62
    profile['PYTHONPATH']=str(BASE/('hooks_'+mode));environment.update(profile)
    sources.update({str(p):digest(p) for p in BASE.rglob('*.py')});frozen_check(sources)
    for name,h in sources.items():
        p=root/'source_snapshot'/Path(name).relative_to('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(name).read_bytes());assert digest(p)==h
    return command,environment,profile,sources
