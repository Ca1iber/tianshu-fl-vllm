# SPDX-License-Identifier: Apache-2.0
"""Current qualified 407-source stack with a geometry-only sampler extension."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_current_topp_grid_parent',BASE.parent/'tianshu_prefill_int8_service_20261021/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(profile)==62 and len(sources)==407 and len(command)==13
    profile['PYTHONPATH']=str(BASE/('hooks_'+mode));environment.update(profile)
    paths=[*BASE.rglob('*.py'),*BASE.rglob('*.sh')]
    sources.update({str(path):digest(path) for path in paths});frozen_check(sources)
    for path,sha in sources.items():
        target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(path).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
