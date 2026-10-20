# SPDX-License-Identifier: Apache-2.0
"""Final future-KV-only profile: measured runtime sources and separate controllers."""
import importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_original_command_20261016'))
from validation_profile import BENCH_SHA,audit_math as original_math,csv_rows,normalized_environment,official_command

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module
parent=load('_future_final_service_profile',BASE.parent/'tianshu_future_kv_service_20261020/candidate_profile.py')
digest,read,frozen_check=parent.digest,parent.read,parent.frozen_check
runtime=load('_future_final_runtime',BASE/'audit_profile.py');audit_runtime=runtime.audit_runtime
legacy=load('_future_dual_official_outputs',BASE.parent/'tianshu_long_segments_validation_20261020/validation_profile_long.py')


def prepare(root,screen_root):
    gate_path=BASE.parent/'tianshu_future_kv_results_20261020/DIAGNOSTIC_16K_GATE_32g.json'
    gate=read(gate_path);control=read(screen_root/'control/result.json');candidate=read(screen_root/'candidate/result.json')
    assert gate['complete'] and gate['independently_rechecked'] and gate['gain_percent']>=2
    assert gate['input_tokens']==16384 and gate['requests_per_round']==64 and gate['excluded_rounds']==[1]
    assert gate['frozen_source_count']==261 and gate['profile_environment_count']==52
    assert control['phase']==candidate['phase']=='complete'
    assert control['frozen_sources']==candidate['frozen_sources']==gate['frozen_sources']
    assert min(r['total_tokens_per_second'] for r in candidate['screen_rounds'][1:])>max(r['total_tokens_per_second'] for r in control['screen_rounds'][1:])
    assert control['spread_percent']<1 and candidate['spread_percent']<1
    short=read(BASE.parent/'tianshu_future_kv_results_20261020/DIAGNOSTIC_4K_GATE_32g.json')
    assert short['complete'] and short['gain_percent']>=-1
    meta_path=BASE.parent/'tianshu_decode_metadata_results_20261020/DIAGNOSTIC_4K_GATE_32g.json'
    metadata=read(meta_path);assert metadata['complete'] and metadata['gain_percent']<.5
    frozen_check(gate['frozen_sources'])
    command,environment,profile,sources=parent.prepare(root,'candidate')
    assert sources==gate['frozen_sources'] and len(sources)==261 and len(profile)==52
    assert command==candidate['server_command']==control['server_command']==official_command()
    expected={k:v.replace(candidate['output_root']+'/',str(root)+'/') for k,v in candidate['profile_environment'].items()}
    assert profile==expected and 'TIANSHU_DECODE_METADATA_MODE' not in profile
    paths=[BASE/'run_validation.py',BASE/'framework_profile.py',BASE/'audit_profile.py',BASE/'batch_policy.json',BASE/'run_math_and_official.sh',
        BASE.parent/'tianshu_original_command_20261016/validation_profile.py',
        BASE.parent/'tianshu_long_segments_validation_20261020/validation_profile_long.py']
    outer={str(p):digest(p) for p in paths}
    (root/'controller_sources.json').write_text(json.dumps(outer,indent=2)+'\n')
    for name,sha in outer.items():
        target=root/'controller_source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes());assert digest(target)==sha
    decision=dict(final_profile_kind='future_output_reservation_only',runtime_source_count=261,environment_count=52,
        independent_16K_screen_gain_percent=gate['gain_percent'],metadata_4K_gain_percent=metadata['gain_percent'],
        metadata_not_included=True,earlier_4K_screen_was260_sources=True,final_same_source_official_4K_required=True,
        fresh_Math_required=True,no_previous_Math_reused=True,screen_gate_sha256=digest(gate_path),metadata_gate_sha256=digest(meta_path),
        runtime_sources_separate_from_controller_sources=True)
    (root/'final_profile_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    return command,environment,profile,sources


def audit_math(root,state,profile,write=True):
    result=original_math(root,state,profile,write=False)
    result.update(final_profile_kind='future_output_reservation_only',controller_sources=state['controller_sources'],
        runtime_source_count=261,environment_count=52,metadata_fusion_not_included=True,
        fresh_Math_for_final_runtime_profile=True,previous_accuracy_not_reused=True)
    if write:(root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def audit_benchmark(root,state,profile,quality,write=True):
    assert state['controller_sources']==quality['controller_sources']
    result=legacy.audit_benchmark(root,state,profile,quality,write=False)
    result.update(final_profile_kind='future_output_reservation_only',controller_sources=state['controller_sources'],
        runtime_source_count=261,environment_count=52,metadata_fusion_not_included=True)
    if write:(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
