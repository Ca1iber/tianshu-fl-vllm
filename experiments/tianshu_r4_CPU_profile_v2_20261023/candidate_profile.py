# SPDX-License-Identifier: Apache-2.0
import importlib.util,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
s=importlib.util.spec_from_file_location('_r4_CPU_actual_model',EXP/'tianshu_W4_LMpad_service_v2_20261023/candidate_profile.py');parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,unused_reference):
 command,env,profile,sources=parent.prepare(root,'candidate');manifest={}
 for row in (EXP/'tianshu_W4_Down64_LM1to64_release_20261023/MODEL_SOURCES.sha256').read_text().splitlines():
  h,p=row.split('  ',1);manifest[p]=h
 assert sources==manifest and len(sources)==944 and len(command)==13 and len(profile)==62
 for p in [*BASE.rglob('*.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/metadata.py')]:sources[str(p)]=digest(p)
 return command,env,profile,sources
