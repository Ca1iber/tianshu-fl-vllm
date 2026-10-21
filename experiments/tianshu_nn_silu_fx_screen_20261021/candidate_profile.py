# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
PROTO=BASE.parent/'tianshu_nn_silu_fx_service_20261021'
spec=importlib.util.spec_from_file_location('_fx_screen_service_profile',PROTO/'candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    gate=read(BASE.parent/'tianshu_nn_silu_fx_integration_recovery_20261021/INTEGRATION_RECOVERY_GATE_32g.json');assert gate['complete'] and gate['diagnostic_integration_only'] and gate['all_42_real_M64_fused_captures_and_replays_verified']
    assert all(digest(p)==h for p,h in gate['frozen_sources'].items())
    assert gate['original_experiment_passed'] is False and gate['post_ready_autotune']==2
    command,environment,profile,sources=parent.prepare(root,mode)
    assert len(profile)==62 and len(command)==13
    profile['PYTHONPATH']=str(BASE/('hooks_'+mode));environment.update(profile)
    paths=[*BASE.rglob('*.py'),BASE/'PLAN_32g.md',BASE/'run_screen.sh',BASE.parent/'tianshu_nn_silu_fx_integration_recovery_20261021/INTEGRATION_RECOVERY_GATE_32g.json',Path('/usr/local/lib/python3.12/site-packages/flag_gems/ops/index.py'),Path('/usr/local/lib/python3.12/site-packages/flag_gems/utils/code_cache.py')]
    sources.update({str(p):digest(p) for p in paths});frozen_check(sources)
    for p,h in sources.items():
        dest=root/'source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes());assert digest(dest)==h
    return command,environment,profile,sources
