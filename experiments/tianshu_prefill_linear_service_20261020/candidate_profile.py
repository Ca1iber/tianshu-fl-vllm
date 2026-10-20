# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_prefill_linear_cold_parent',BASE.parent/'tianshu_cold_start_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'gpu4')
    assert len(profile)==59 and len(sources)==330
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_PREFILL_LINEAR_MODE=mode,TIANSHU_PREFILL_LINEAR_RECORD_DIR=str(root/'prefill_linear_records'))
    assert len(profile)==61;environment.update(profile)
    gate=read(BASE.parent/'tianshu_gateup_bf16_dispatch_20261020/CONTIGUOUS_COPY_GATE_32g.json')
    assert gate['complete'] and not gate['failures']
    for path in [*BASE.rglob('*.py'),*[Path(p) for p in gate['frozen_sources']],BASE.parent/'tianshu_gateup_bf16_dispatch_20261020/CONTIGUOUS_COPY_GATE_32g.json']:
        sources[str(path)]=digest(path)
    frozen_check(sources)
    for name,sha in sources.items():
        target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
