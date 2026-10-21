# SPDX-License-Identifier: Apache-2.0
"""Both arms retain qualified450 M64; only new dynamic range fusion varies."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_range_profile_qualified450',BASE.parent/'tianshu_nn_silu_fx_screen_20261021/candidate_profile.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
 assert mode in ('control','candidate')
 command,environment,profile,sources=parent.prepare(root,'candidate');assert len(sources)==450 and len(profile)==62 and len(command)==13
 old=dict(sources);old_env=dict(profile)
 selection=read(BASE/'selected_plan.json');component=Path(selection['component_gate']);gate=read(component)
 assert gate['complete'] and gate['performance_valid'] and digest(component)==selection['component_gate_sha256']
 assert all(digest(p)==h for p,h in gate['source_hashes'].items())
 sources.update(gate['source_hashes'])
 paths=[*BASE.rglob('*.py'),BASE/'PLAN_32g.md',BASE/'selected_plan.json',BASE/'run_pair.sh',component,BASE.parent/'tianshu_nn_silu_large_tiles_v2_20261021/kernel.py',BASE.parent/'tianshu_m_range_mlp42_v2_20261021/inductor_activation.py']
 sources.update({str(p):digest(p) for p in paths});frozen_check(sources);assert all(sources[p]==h for p,h in old.items())
 profile['PYTHONPATH']=str(BASE/('hooks_'+mode));environment.update(profile);assert all(profile[k]==v for k,v in old_env.items() if k!='PYTHONPATH')
 for p,h in sources.items():
  dest=root/'source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes());assert digest(dest)==h
 return command,environment,profile,sources
