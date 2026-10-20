# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_metadata_future_profile',BASE.parent/'tianshu_future_kv_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    metadata=BASE.parent/'tianshu_decode_metadata_20261020';gpu=read(metadata/'GPU_GATE_32g.json')
    assert gpu['complete'] and gpu['GPU_cases']==14
    frozen_check(gpu['frozen_sources'])
    command,environment,profile,sources=parent.prepare(root,'candidate')
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_DECODE_METADATA_MODE=mode,
                   TIANSHU_DECODE_METADATA_RECORD_DIR=str(root/'decode_metadata_records'))
    assert len(profile)==54 and profile['TIANSHU_FUTURE_KV_MODE']=='candidate'
    environment.update(profile)
    paths=[*BASE.rglob('*.py'),*metadata.glob('*.py'),metadata/'CPU_GATE_32g.json',metadata/'GPU_GATE_32g.json',
           *[Path(p) for p in gpu['frozen_sources']]]
    for path in paths:sources[str(path)]=digest(path)
    frozen_check(sources)
    for name,sha in sources.items():
        target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
