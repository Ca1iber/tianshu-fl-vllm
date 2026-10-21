# SPDX-License-Identifier: Apache-2.0
"""Current model precision and scheduler, only the sampler geometry changes."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_grid_service_current_parent',BASE.parent/'tianshu_prefill_int8_service_20261021/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(profile)==62 and len(sources)==407 and len(command)==13
    selection=read(BASE/'selected_plan.json');gate_path=Path(selection['API_gate']);gate=read(gate_path)
    assert digest(gate_path)==selection['API_gate_sha256'] and gate['complete'] and gate['independently_CPU_recomputed']
    assert gate['full_FP32_outputs_reconstructed']==600 and gate['all_per_row_original_contract_passed']
    assert gate['RNG_cases']==33 and gate['same_graph_lifecycle_cases']==8 and gate['strict_false_rows_all_equal_baseline']
    profile['PYTHONPATH']=str(BASE/('hooks_'+mode));environment.update(profile)
    paths=[*BASE.rglob('*.py'),*BASE.rglob('*.sh'),BASE/'selected_plan.json',gate_path,BASE.parent/'tianshu_topp_grid_current_api_20261021/verify_gpu.py',*[Path(p) for p in gate['frozen_sources']]]
    sources.update({str(path):digest(path) for path in paths});frozen_check(sources)
    for path,sha in sources.items():
        target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(path).read_bytes());assert digest(target)==sha
    gate_file=BASE/"CPU_GATE_32g.json"
    if gate_file.is_file():
        target=root/"source_snapshot"/gate_file.relative_to("/");target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(gate_file.read_bytes())
    return command,environment,profile,sources
