# SPDX-License-Identifier: Apache-2.0
"""The qualified precision stack with explicit budget guards and startup shapes."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_int8_prefill_budget_qualified_parent',BASE.parent/'tianshu_prefill_int8_service_20261021/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate');assert len(profile)==62 and len(sources)==407
    profile.update(PYTHONPATH=str(BASE/('hooks_'+mode)),TIANSHU_DENSE_BUDGET_MODE=mode,TIANSHU_DENSE_BUDGET_RECORD_DIR=str(root/'dense_budget_records'))
    assert len(profile)==64;environment.update(profile)
    budget=BASE.parent/'tianshu_dense_budget8192_v2_20261021';nn=BASE.parent/'tianshu_prefill_budget8192_nn_20261021'
    selection=read(BASE/'selected_plan.json');large=Path(selection['large_component_gate']);gate=read(large)
    assert digest(large)==selection['large_component_gate_sha256'] and gate['complete'] and gate['independently_CPU_rechecked']
    paths=[*BASE.rglob('*.py'),*BASE.rglob('*.sh'),BASE/'selected_plan.json',*nn.rglob('*.py'),*budget.rglob('*.py'),budget/'batch_policy.json',large,large.parent/'verify_gpu.py',*[Path(p) for p in gate['frozen_sources']]]
    sources.update({str(p):digest(p) for p in paths});frozen_check(sources)
    for path,sha in sources.items():
        target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
