# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
rms=load('_RMS_LM_model_RMS_parent',EXP/'tianshu_RMS_A8_GU_service_v2_20261023/candidate_profile.py');lm=load('_RMS_LM_model_LM_parent',EXP/'tianshu_LM_pad64_service_20261022/candidate_profile.py');digest,read,frozen_check=rms.digest,rms.read,rms.frozen_check

def prepare(root,mode='candidate',check_gate=True):
 assert mode in ('control','candidate');command,environment,profile,sources=rms.prepare(root,mode);lc,le,lp,ls=lm.prepare(root,mode);assert command==lc and len(sources)==916 and len(ls)==925 and {k:v for k,v in profile.items() if k!='PYTHONPATH'}=={k:v for k,v in lp.items() if k!='PYTHONPATH'} and environment['TRITON_CACHE_DIR']==le['TRITON_CACHE_DIR'];parents=(dict(sources),dict(ls));assert all(sources[p]==h for p,h in ls.items() if p in sources);sources.update(ls)
 paths=[*BASE.rglob('*.py'),BASE/'PLAN_32g.md'];sources.update({str(p):digest(p) for p in paths});frozen_check(sources);assert all(sources[p]==h for parent in parents for p,h in parent.items())
 if check_gate:gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'] and gate['sources']==sources;sources[str(BASE/'CPU_GATE_32g.json')]=digest(BASE/'CPU_GATE_32g.json')
 profile['PYTHONPATH']=str(BASE/f'hooks_{mode}');environment.update(profile)
 for p,h in sources.items():
  t=root/'source_snapshot'/Path(p).relative_to('/');t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(Path(p).read_bytes());assert digest(t)==h
 return command,environment,profile,sources
