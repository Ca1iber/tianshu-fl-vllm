# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
s=importlib.util.spec_from_file_location('_RMS_A8_GU_r3',EXP/'tianshu_native_int8_attention_service_20261022/candidate_profile.py');parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode='candidate',check_gate=True):
 assert mode in ('control','candidate');command,environment,profile,sources=parent.prepare(root,'candidate');assert len(sources)==903;old=dict(sources)
 paths=[*BASE.rglob('*.py'),BASE/'PLAN_32g.md',EXP/'tianshu_IR_RMS_A8_20261023/kernel.py',EXP/'tianshu_IR_RMS_A8_20261023/INDEPENDENT_CPU_GATE_32g.json',EXP/'tianshu_IR_RMS_A8_chain_20261023/INDEPENDENT_CPU_GATE_32g.json',EXP/'tianshu_RMS_A8_FX_inventory_20261023/CPU_INVENTORY_GATE_32g.json']
 sources.update({str(p):digest(p) for p in paths});frozen_check(sources);assert all(sources[p]==h for p,h in old.items())
 if check_gate:gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'] and sources==gate['sources'];sources[str(BASE/'CPU_GATE_32g.json')]=digest(BASE/'CPU_GATE_32g.json')
 profile['PYTHONPATH']=str(BASE/f'hooks_{mode}');environment.update(profile)
 for p,h in sources.items():
  t=root/'source_snapshot'/Path(p).relative_to('/');t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(Path(p).read_bytes());assert digest(t)==h
 return command,environment,profile,sources
