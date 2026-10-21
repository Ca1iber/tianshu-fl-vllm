# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_qkv_prefill_current_407_profile',BASE.parent/'tianshu_prefill_int8_service_20261021/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(sources)==407 and len(profile)==62 and len(command)==13
    profile['PYTHONPATH']=str(BASE/('hooks_'+mode));environment.update(profile)
    selected=read(BASE/'selected_plan.json');gate_path=Path(selected['component_gate']);gate=read(gate_path)
    assert selected['component_gate_sha256']==digest(gate_path) and gate['complete'] and gate['independently_CPU_rechecked']
    assert gate['original_GPU_run_remains_failed'] and gate['partial_scope']=='completed_QKV_only'
    assert selected['kernel_sha256']==digest(selected['kernel_source']) and len(selected['configurations'])==6
    paths=[*BASE.rglob('*.py'),BASE/'selected_plan.json',BASE/'CHECKPOINT_SIGNATURES_32g.json',gate_path,gate_path.parent/'verify_gpu.py',*[Path(p) for p in gate['frozen_sources']]]
    sources.update({str(path):digest(path) for path in paths});frozen_check(sources)
    for path,sha in sources.items():
        target=Path(root)/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
