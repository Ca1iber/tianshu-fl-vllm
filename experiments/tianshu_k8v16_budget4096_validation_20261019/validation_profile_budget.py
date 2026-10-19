# SPDX-License-Identifier: Apache-2.0
"""Bind fresh Math/formal runs to the completed, source-identical scheduler screen."""
import importlib.util
from pathlib import Path
from audit_outputs import BENCH_SHA,audit_math,audit_benchmark,csv_rows,normalized_environment
BASE=Path(__file__).resolve().parent
p=BASE.parent/'tianshu_k8v16_budget4096_20261019/candidate_profile.py'
s=importlib.util.spec_from_file_location('_budget4096_screen_profile',p);parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
prepare_screen,digest,read,frozen_check=parent.prepare,parent.digest,parent.read,parent.frozen_check
p=BASE.parent/'tianshu_k8v16_budget4096_20261019/audit_profile.py'
s=importlib.util.spec_from_file_location('_budget4096_validation_runtime_audit',p);audit=importlib.util.module_from_spec(s);s.loader.exec_module(audit);audit_runtime=audit.audit_runtime

def prepare(root,screen_root):
    confirm=read(screen_root/'comparison.json');control=read(screen_root/'control/result.json');candidate=read(screen_root/'candidate/result.json')
    assert confirm['complete'] and confirm['gain_percent']>=2
    assert control['phase']==candidate['phase']=='complete' and confirm['rounds_per_mode']==3 and confirm['excluded_rounds']==[1]
    assert min(r['total_tokens_per_second'] for r in candidate['short_4k_rounds'][1:])>max(r['total_tokens_per_second'] for r in control['short_4k_rounds'][1:])
    assert control['spread_percent']<1 and candidate['spread_percent']<1
    assert candidate['frozen_sources']==control['frozen_sources']==confirm['frozen_sources']
    frozen_check(confirm['frozen_sources'])
    command,environment,profile,sources=prepare_screen(root,'candidate')
    assert sources==confirm['frozen_sources']
    assert command==candidate['server_command']==control['server_command']+['--max-num-batched-tokens','4096']
    expected={k:v.replace(candidate['output_root']+'/',str(root)+'/') for k,v in candidate['profile_environment'].items()}
    assert profile==expected and len(profile)==41 and profile['TIANSHU_K8V16_SERVICE_MODE']=='candidate'
    gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'] and gate['no_new_quality_claim']
    for name,h in gate['sources'].items():assert digest(BASE/name)==h,name
    for name in ('validation_profile_budget.py','audit_outputs.py','run_validation.py','verify_validation_downloaded.py','check_cpu.py','CPU_GATE_32g.json'):
        p=BASE/name;sources[str(p)]=digest(p)
    frozen_check(sources)
    for path,h in sources.items():
        dest=root/'source_snapshot'/Path(path).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(path).read_bytes());assert digest(dest)==h
    return command,environment,profile,sources
