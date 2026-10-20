# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_kv_empty_qualified_profile',BASE.parent/'tianshu_lm_precision_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(profile)==62 and len(sources)==359
    profile['PYTHONPATH']=str(BASE/('hooks_'+mode));environment.update(profile)
    component=BASE.parent/'tianshu_kv_empty_allocate_20261021'
    gates=[component/'CPU_GATE_32g.json',component/'CPU_GUARD_GATE_32g.json']
    paths=[*BASE.rglob('*.py'),*gates]
    for path in gates:
        gate=read(path);assert gate['complete'];paths.extend(Path(p) for p in gate['sources'])
    sources.update({str(p):digest(p) for p in paths});frozen_check(sources)
    for path,sha in sources.items():
        target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
