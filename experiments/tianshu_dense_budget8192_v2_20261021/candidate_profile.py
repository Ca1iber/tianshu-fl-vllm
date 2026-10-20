# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_dense_budget_qualified_LM',BASE.parent/'tianshu_lm_precision_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(sources)==359 and len(profile)==62
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_DENSE_BUDGET_MODE=mode,TIANSHU_DENSE_BUDGET_RECORD_DIR=str(root/'dense_budget_records'))
    assert len(profile)==64;environment.update(profile)
    paths=[*BASE.rglob('*.py'),BASE/'batch_policy.json']
    for path in paths:sources[str(path)]=digest(path)
    frozen_check(sources)
    for name,sha in sources.items():
        target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
