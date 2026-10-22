# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_native_profile_qualified_range',BASE.parent/'tianshu_m_range_prefill_service_v2_20261021/candidate_profile.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
CACHE_GATE=BASE.parent/'tianshu_prefill_control_binary_all_20261022/QUALIFIED_MODEL_CACHE_32g.json'
def prepare(root,mode,check_gate=True):
    assert mode in ('control','candidate')
    command,environment,profile,sources=parent.prepare(root,'candidate');assert len(command)==13 and len(profile)==62 and len(sources)==471
    inherited=dict(sources);inherited_env=dict(profile)
    holdout=BASE.parent/'tianshu_native_int8_mlp42_20261022';gate=read(holdout/'INDEPENDENT_CPU_GATE_32g.json');assert gate['complete'] and gate['raw_result_sha256']==digest(holdout/'GPU_GATE_32g.json')
    cache=read(CACHE_GATE);assert cache['complete'] and cache['total_strict_binary_points']==24
    paths=[*BASE.rglob('*.py'),BASE/'PLAN_32g.md',BASE/'run_smoke.sh',CACHE_GATE,holdout/'INDEPENDENT_CPU_GATE_32g.json',holdout/'GPU_GATE_32g.json',holdout/'KEY_TIMINGS_32g.csv',BASE.parent/'tianshu_native_int8_prefill_v2_20261022/kernel.py',BASE.parent/'tianshu_native_int8_prefill_v2_20261022/GPU_GATE_32g.json']
    sources.update({str(path):digest(path) for path in paths});sources.update(cache['files']);frozen_check(sources);assert all(sources[path]==value for path,value in inherited.items())
    if check_gate:
        own=read(BASE/'CPU_GATE_32g.json');assert own['complete'] and own['sources']==sources
        sources[str(BASE/'CPU_GATE_32g.json')]=digest(BASE/'CPU_GATE_32g.json')
    profile['PYTHONPATH']=str(BASE/('hooks_'+mode));environment.update(profile);environment['TRITON_CACHE_DIR']=cache['qualified_model_cache_directory']
    assert all(profile[key]==value for key,value in inherited_env.items() if key!='PYTHONPATH')
    for path,value in sources.items():
        target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert digest(target)==value
    return command,environment,profile,sources
