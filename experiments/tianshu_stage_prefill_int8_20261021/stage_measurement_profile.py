# SPDX-License-Identifier: Apache-2.0
"""Qualified407/62env plus outside-Graph observation; no numerical replacement."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
parent=load('_stage_current_prefill_int8_profile',BASE.parent/'tianshu_prefill_int8_service_20261021/candidate_profile.py')
audit=load('_stage_current_prefill_int8_audit',BASE.parent/'tianshu_prefill_int8_service_20261021/audit_profile.py')
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root):
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(profile)==62 and len(sources)==407
    original_sources=dict(sources);original_profile=dict(profile)
    profile['PYTHONPATH']=str(BASE/'hooks');environment.update(profile)
    assert {k:v for k,v in profile.items() if k!='PYTHONPATH'}=={k:v for k,v in original_profile.items() if k!='PYTHONPATH'}
    paths=[*BASE.rglob('*.py'),BASE/'run_profile.sh',BASE/'PLAN_32g.md',
           BASE.parent/'tianshu_stage_current_20261020/hooks/sitecustomize.py',
           BASE.parent/'tianshu_stage_current_20261020/run_profile.py',
           BASE.parent/'tianshu_prefill_int8_official_results_20261021/RUNTIME_BINDING_GATE_32g.json']
    sources.update({str(p):digest(p) for p in paths});frozen_check(sources)
    assert all(sources[path]==sha for path,sha in original_sources.items())
    for path,sha in sources.items():
        dest=root/'source_snapshot'/Path(path).relative_to('/')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(path).read_bytes());assert digest(dest)==sha
    return command,environment,profile,sources

def audit_runtime(root,state,official=False):
    assert len(state['profile_environment'])==62
    assert state['profile_environment']['PYTHONPATH']==str(BASE/'hooks')
    result=audit.audit_runtime(root,state,official=official)
    assert result['actual_new_prefill_path_all_42_GateUp_verified']
    log=(root/'server.log').read_text(errors='replace')
    assert log.count('Application startup complete.')==1
    tail=log.split('Application startup complete.',1)[1]
    records=[line for line in tail.splitlines() if 'Triton autotuning for function' in line]
    assert not records,records
    result.update(diagnostic_observation_only=True,post_ready_autotuning_records=len(records),
                  full_62_environment_and_original_argv_retained=True,no_precision_or_schedule_change=True)
    return result
