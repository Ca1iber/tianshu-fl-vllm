# SPDX-License-Identifier: Apache-2.0
"""Both arms use qualified K8V16; candidate explicitly changes scheduler token budget."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
p=BASE.parent/'tianshu_k8v16_budget4096_20261019/candidate_profile.py'
s=importlib.util.spec_from_file_location('_k8v16_budget_parent',p);parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def budget_command(command,mode):
    assert mode in ('control','candidate') and command.count('--max-num-batched-tokens')==1
    assert command[-2:]==['--max-num-batched-tokens','4096']
    return list(command) if mode=='control' else list(command[:-1])+['8192']

def prepare(root,mode):
    gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete']
    for n,h in gate['sources'].items():assert digest(BASE/n)==h,n
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(profile)==41 and profile['TIANSHU_K8V16_SERVICE_MODE']=='candidate'
    command=budget_command(command,mode)
    for n in ('candidate_profile.py','audit_profile.py','run_service_pair.py','run_service_trial.py','check_cpu.py','verify_service_downloaded.py','CPU_GATE_32g.json'):
        p=BASE/n;sources[str(p)]=digest(p)
    frozen_check(sources)
    for n,h in sources.items():
        p=root/'source_snapshot'/Path(n).relative_to('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(n).read_bytes());assert digest(p)==h
    return command,environment,profile,sources
