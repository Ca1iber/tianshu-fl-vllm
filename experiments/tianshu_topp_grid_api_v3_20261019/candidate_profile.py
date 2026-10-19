# SPDX-License-Identifier: Apache-2.0
"""Current best profile with a separate Top-P grid switch, no service argv edits."""
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_gemm_topp_geo_20261016'))
from validation_profile_new import prepare as prepare_base,digest,frozen_check,read


def prepare(root,mode):
    assert mode in ('control','candidate')
    gate=read(BASE/'CPU_GATE_32g.json')
    assert gate['complete'] and gate['CPU_only'] and gate['routing_cases']==30
    assert gate['grid_buffer_cases']==96 and gate['shared_scratch_buffers']==1
    for name,expected in gate['sources'].items():assert digest(BASE/name)==expected,name
    command,env,profile,sources=prepare_base(root,Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
    assert len(profile)==38
    profile.update(PYTHONPATH=str(BASE/'hooks'),TIANSHU_TOPP_GRID_CANDIDATE='1' if mode=='candidate' else '0',
                   TIANSHU_TOPP_GRID_RECORD_DIR=str(root/'grid_topp_records'))
    assert len(profile)==40;env.update(profile)
    names=('candidate_profile.py','grid_route.py','kernel_route.py','hooks/sitecustomize.py','probe_api.py',
           'run_api_probe.py','run_service_trial.py','run_service_pair.py','audit_profile.py','check_cpu.py','CPU_GATE_32g.json')
    for name in names:sources[str(BASE/name)]=digest(BASE/name)
    p=BASE.parent/'tianshu_topp_grid_20261019/router.py';sources[str(p)]=digest(p)
    frozen_check(sources)
    for path,expected in sources.items():
        target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(path).read_bytes());assert digest(target)==expected
    return command,env,profile,sources
