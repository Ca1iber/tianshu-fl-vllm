# SPDX-License-Identifier: Apache-2.0
"""Both arms keep original serve argv and qualified K8V16; only guarded default differs."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
p=BASE.parent/'tianshu_k8v16_pipeline_20261019/candidate_profile.py'
s=importlib.util.spec_from_file_location('_K8V16_original_argv_default_parent',p);parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'] and gate['actual_installed_default_setting_AST_executed']
    for n,h in gate['sources'].items():assert digest(BASE/n)==h,n
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(profile)==41 and profile['TIANSHU_K8V16_SERVICE_MODE']=='candidate'
    assert '--max-num-batched-tokens' not in command
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_BUDGET_DEFAULT_MODE=mode,TIANSHU_BUDGET_DEFAULT_RECORD_DIR=str(root/'default_budget_records'))
    assert len(profile)==43;environment.update(profile)
    for n in ('candidate_profile.py','audit_profile.py','default_policy.py','hooks/sitecustomize.py','run_service_pair.py','run_service_trial.py','check_cpu.py','verify_service_downloaded.py','CPU_GATE_32g.json'):
        p=BASE/n;sources[str(p)]=digest(p)
    p=Path('/usr/local/lib/python3.12/site-packages/vllm/engine/arg_utils.py')
    assert digest(p)=='f1a32e2207e9ed2f0d954e3e6457effca17ca907cb0cf17bba2ca8bde855a719';sources[str(p)]=digest(p)
    frozen_check(sources)
    for n,h in sources.items():
        p=root/'source_snapshot'/Path(n).relative_to('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(n).read_bytes());assert digest(p)==h
    return command,environment,profile,sources
