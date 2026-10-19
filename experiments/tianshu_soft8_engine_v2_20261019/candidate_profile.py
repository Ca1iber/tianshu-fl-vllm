# SPDX-License-Identifier: Apache-2.0
"""Replay the qualified profile with the original service argv and byte KV switch."""
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_gemm_topp_geo_20261016'))
from validation_profile_new import prepare as prepare_base,digest,frozen_check,read

def prepare(root,mode):
    assert mode in ('control','candidate')
    cpu=read(BASE/'CPU_GATE_32g.json');assert cpu['complete']
    for name,h in cpu['sources'].items():assert digest(BASE/name)==h
    gpu=read(BASE/'GPU_GATE_32g.json');assert gpu['complete'] and gpu['actual_builder_writer_forward_cases']==21
    frozen_check(gpu['frozen_sources'])
    command,env,profile,sources=prepare_base(root,Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
    assert len(profile)==38
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_SOFT8_SERVICE_MODE=mode,
                   TIANSHU_SOFT8_RECORD_DIR=str(root/'soft8_engine_records'))
    assert len(profile)==40;env.update(profile);sources.update(gpu['frozen_sources'])
    for name in ('candidate_profile.py','audit_profile.py','engine_hooks.py','hooks/sitecustomize.py',
                 'ENGINE_BINDINGS_32g.json','GPU_GATE_32g.json','run_service_trial.py','run_service_pair.py'):
        sources[str(BASE/name)]=digest(BASE/name)
    sources.update(read(BASE/'ENGINE_BINDINGS_32g.json'));frozen_check(sources)
    for name,h in sources.items():
        target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes());assert digest(target)==h
    return command,env,profile,sources
