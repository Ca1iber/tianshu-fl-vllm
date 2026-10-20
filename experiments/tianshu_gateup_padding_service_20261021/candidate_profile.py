# SPDX-License-Identifier: Apache-2.0
"""Qualified LM BF16 profile; one independent GateUp tile-padding flag."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_gateup_padding_qualified_parent',BASE.parent/'tianshu_lm_precision_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(profile)==62 and len(sources)==359
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_TILE_PADDING_MODE=mode);assert len(profile)==63;environment.update(profile)
    pad=BASE.parent/'tianshu_dense_tile_padding_20261021'
    paths=[*BASE.rglob('*.py'),pad/'kernel_padded.py',pad/'GPU_GATE_32g.json']
    initial=set(sources)
    for path in paths:sources[str(path)]=digest(path)
    assert len(sources)==359+len(set(str(p) for p in paths)-initial)
    frozen_check(sources)
    for name,sha in sources.items():
        dst=root/'source_snapshot'/Path(name).relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(name).read_bytes());assert digest(dst)==sha
    return command,environment,profile,sources
