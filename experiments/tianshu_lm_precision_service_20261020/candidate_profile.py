# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_LM_precision_prefill_parent',BASE.parent/'tianshu_prefill_linear_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(profile)==61 and len(sources)==348
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_LM_PRECISION_MODE=mode);assert len(profile)==62;environment.update(profile)
    paths=[*BASE.rglob('*.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/utils.py'),Path('/usr/local/lib/python3.12/site-packages/vllm_fl/dispatch/io_common.py')]
    for path in paths:sources[str(path)]=digest(path)
    frozen_check(sources)
    for name,sha in sources.items():
        dst=root/'source_snapshot'/Path(name).relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(name).read_bytes());assert digest(dst)==sha
    return command,environment,profile,sources
