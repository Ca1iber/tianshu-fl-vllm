# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_pair_future_profile',BASE.parent/'tianshu_future_kv_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    component=BASE.parent/'tianshu_k8v16_pair_load_20261020';gpu=read(component/'GPU_GATE_32g.json')
    assert gpu['complete'] and gpu['independently_rechecked'] and gpu['full_GPU_FP32_output_checks']==96
    assert gpu['all_candidate_output_tensors_bitwise_equal_to_current']
    frozen_check(gpu['frozen_sources'])
    command,environment,profile,sources=parent.prepare(root,'candidate')
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_PAIR_LOAD_MODE=mode,
                   TIANSHU_PAIR_LOAD_RECORD_DIR=str(root/'pair_load_records'))
    assert len(profile)==54 and profile['TIANSHU_FUTURE_KV_MODE']=='candidate'
    environment.update(profile)
    paths=[*BASE.rglob('*.py'),*component.glob('*.py'),component/'CPU_GATE_32g.json',component/'GPU_GATE_32g.json',
           *[Path(p) for p in gpu['frozen_sources']]]
    for path in paths:sources[str(path)]=digest(path)
    frozen_check(sources)
    for name,sha in sources.items():
        target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
