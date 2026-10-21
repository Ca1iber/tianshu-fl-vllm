# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_topp_warp_current_prefill_profile',BASE.parent/'tianshu_prefill_int8_service_20261021/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate');assert len(profile)==62 and len(sources)==407 and len(command)==13
    profile['PYTHONPATH']=str(BASE/('hooks_'+mode));environment.update(profile)
    selected=read(BASE/'selected_plan.json');gate_path=Path(selected['component_gate']);gate=read(gate_path)
    assert gate['complete'] and gate['CPU_reverified'] and selected['component_gate_sha256']==digest(gate_path)
    assert selected['exploratory_non_bitwise_candidate'] and not gate['bitwise_native_all_cases_by_config']['warp16']
    paths=[*BASE.rglob('*.py'),*BASE.glob('*.sh'),BASE/'PLAN.md',BASE/'selected_plan.json',gate_path,gate_path.parent.parent/'verify_cpu.py',Path(selected['TV_summary']),Path(selected['layout_equivalence']),*[Path(p) for p in gate['frozen_sources']]]
    sources.update({str(p):digest(p) for p in paths});frozen_check(sources)
    for path,sha in sources.items():
        target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert digest(target)==sha
    cpu_path=BASE/"CPU_GATE_32g.json"
    if cpu_path.exists():
        cpu=read(cpu_path);assert cpu["complete"] and cpu["profiles"][0]["sources"]==sources and len(sources)==cpu["runtime_source_count"]
        target=root/"source_snapshot"/cpu_path.relative_to("/");target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(cpu_path.read_bytes());assert digest(target)==digest(cpu_path)
    return command,environment,profile,sources
