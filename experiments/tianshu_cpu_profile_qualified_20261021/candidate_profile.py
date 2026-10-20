# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_qualified359_profile_parent',BASE.parent/'tianshu_lm_precision_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,unused_reference):
    command,env,profile,sources=parent.prepare(root,'candidate')
    assert len(command)==13 and len(profile)==62 and len(sources)==359
    for path in [*BASE.rglob('*.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/metadata.py')]:sources[str(path)]=digest(path)
    return command,env,profile,sources
