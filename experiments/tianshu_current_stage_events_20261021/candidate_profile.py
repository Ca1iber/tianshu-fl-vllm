# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_current471_stage_parent',BASE.parent/'tianshu_m_range_prefill_service_v2_20261021/candidate_profile.py');parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,unused_reference):
 command,env,profile,sources=parent.prepare(root,'candidate')
 original=read(BASE.parent/'tianshu_m_range_prefill_official_results_20261021/OFFICIAL_GATE_32g.json')['model_frozen_sources'];assert len(sources)==471 and sources==original and len(profile)==62 and len(command)==13
 for p in [*BASE.rglob('*.py'),*BASE.glob('*.sh'),BASE/'PLAN_32g.md']:sources[str(p)]=digest(p)
 return command,env,profile,sources
