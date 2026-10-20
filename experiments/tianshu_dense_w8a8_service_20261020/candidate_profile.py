# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_dense_existing_pair_profile',BASE.parent/'tianshu_k8v16_pair_service_20261020/candidate_profile.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check

def prepare(root,mode):
    assert mode in ('control','candidate')
    component=BASE.parent/'tianshu_dense_w8a8_20261020';gate=read(component/'GPU_GATE_32g.json')
    assert gate['complete'] and gate['independently_CPU_rechecked'] and gate['compiled_cases']==113
    frozen_check(gate['frozen_sources'])
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert len(profile)==54 and len(sources)==282
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_DENSE_INT8_MODE=mode,TIANSHU_DENSE_INT8_RECORD_DIR=str(root/'dense_int8_records'))
    assert len(profile)==56;environment.update(profile)
    native=Path('/usr/local/lib/python3.12/site-packages/vllm')
    paths=[*BASE.rglob('*.py'),component/'kernel.py',component/'GPU_GATE_32g.json',component/'MODEL_INTEGRATION_PLAN_32g.json',
        *[Path(p) for p in gate['frozen_sources']],native/'model_executor/layers/linear.py',native/'model_executor/layers/vocab_parallel_embedding.py',
        native/'model_executor/model_loader/base_loader.py',native/'model_executor/model_loader/default_loader.py',native/'model_executor/model_loader/utils.py',native/'utils/mem_utils.py']
    for path in paths:sources[str(path)]=digest(path)
    frozen_check(sources)
    for name,sha in sources.items():
        target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());assert digest(target)==sha
    return command,environment,profile,sources
