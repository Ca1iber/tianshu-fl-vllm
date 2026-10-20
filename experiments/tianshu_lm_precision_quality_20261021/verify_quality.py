# SPDX-License-Identifier: Apache-2.0
"""Independent complete Math audit and compact score/usage/source archive."""
from collections import Counter
import hashlib,json,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
PIPELINE=BASE.parent/'tianshu_lm_precision_validation_20261020'
sys.path.insert(0,str(PIPELINE))
import framework_profile as profile

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):(BASE/name).write_text(json.dumps(value,indent=2)+'\n')
def signature(runtime):
    dense=runtime['dense_int8_record'];assert dense['load_complete'] and dense['actual_packed_banks']==43
    return sorted([{k:r[k] for k in ('role','prefix','shape','original_BF16_sha256','packed_sha256','scales_sha256','packed_bytes')} for r in dense['banks']],key=lambda r:r['prefix'])
def main():
    root=Path(sys.argv[1]);official=Path(sys.argv[2]);state=read(root/'result.json')
    assert state['phase']=='complete' and state['stage']=='math'
    profile.frozen_check(state['frozen_sources']);profile.frozen_check(state['controller_sources'])
    assert len(state['frozen_sources'])==359 and len(state['profile_environment'])==62
    expected={str(p) for p in [*PIPELINE.glob('*.py'),*PIPELINE.glob('*.sh'),PIPELINE/'batch_policy.json',BASE.parent/'tianshu_original_command_20261016/validation_profile.py',Path(state['screen_root'])/'SCREEN_GATE_32g.json']}
    assert set(state['controller_sources'])==expected
    env=state['profile_environment'];assert env['VLLM_DISABLE_COMPILE_CACHE']=='1' and env['TIANSHU_PREFILL_LINEAR_MODE']=='candidate' and env['TIANSHU_COLD_START_MODE']=='gpu4'
    command=state['evaluation_command']
    def arg(name):assert command.count(name)==1;return command[command.index(name)+1]
    assert arg('--eval-batch-size')=='8' and arg('--timeout')=='3600' and arg('--api-key')=='EMPTY'
    assert json.loads(arg('--generation-config'))==dict(temperature=1.,top_p=.95,max_tokens=32768)
    assert json.loads(arg('--dataset-args'))=={'math_500':{'dataset_id':'/workspace/evalscope-datasets/math_500','subset_list':['Level 3']}}
    runtime=profile.audit_runtime(root,state,official=False);quality=profile.audit_math(root,state,runtime,write=False)
    assert quality==read(root/'accuracy_audit.json') and quality['complete'] and quality['total']==105
    assert runtime['verified_dense_batch']==runtime['verified_batch']==8
    assert runtime['LM_head_precision_in_verified_batch']=='BF16' and quality['actual_C8_LM_BF16_path_verified'] and quality['actual_C8_GateUp_integer_path_verified'] and quality['LM_M64_integer_not_exercised_by_C8_Math']
    assert state['profile_environment']['TIANSHU_LM_PRECISION_MODE']=='candidate'
    assert runtime['actual_new_prefill_path_all_42_GateUp_verified'] and quality['actual_new_prefill_path_all_42_GateUp_verified']
    screen=read(Path(state['screen_root'])/'SCREEN_GATE_32g.json');screened=read(Path(screen['raw_roots']['candidate'])/'result.json')
    banks=signature(runtime);assert banks==signature(screened['profile_audit'])==read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')
    run=next(root.glob('evalscope/*/reports/minicpm/math_500.json')).parents[2]
    predictions=run/'predictions/minicpm/math_500_Level 3.jsonl';reviews=run/'reviews/minicpm/math_500_Level 3.jsonl'
    plines=predictions.read_bytes().splitlines();rlines=reviews.read_bytes().splitlines()
    pd={json.loads(line)['index']:(json.loads(line),line) for line in plines};rd={json.loads(line)['index']:(json.loads(line),line) for line in rlines}
    assert len(plines)==len(rlines)==len(pd)==len(rd)==105 and set(pd)==set(rd)==set(range(105))
    rows=[]
    for index in range(105):
        p,pl=pd[index];r,rl=rd[index];out=p['model_output'];score=r['sample_score']['score']
        assert out['error'] is None and len(out['choices'])==1 and score['status']=='success'
        accuracy=score['value']['accuracy'];assert accuracy in (0,1)
        usage=out['usage'];assert usage['input_tokens']>0 and 0<usage['output_tokens']<=32768 and usage['input_tokens']+usage['output_tokens']==usage['total_tokens']
        rows.append(dict(index=index,accuracy=accuracy,stop_reason=out['choices'][0]['stop_reason'],usage=usage,
            original_prediction_line_SHA256_without_newline=hashlib.sha256(pl).hexdigest(),original_review_line_SHA256_without_newline=hashlib.sha256(rl).hexdigest()))
    assert sum(r['accuracy'] for r in rows)==quality['correct'] and state['accuracy_passed']==quality['official_requirement_passed']==(quality['accuracy']>.95)
    for relative,metadata in quality['raw_files'].items():assert sha(root/relative)==metadata['sha256'] and (root/relative).stat().st_size==metadata['bytes']
    route=runtime['prefill_linear_route'];actual=[r for r in route['calls'].values() if r['stage']=='ready' and not r['actually_inside_capture'] and r['M']>=128]
    assert len({r['prefix'] for r in actual})==42 and all(r['implementation']=='linear_contiguous' for r in actual)
    official_verified=False;phase=None
    if (official/'state.json').exists():
        live=read(official/'state.json');phase=live['phase']
        assert live['stage']=='official' and live['frozen_sources']==state['frozen_sources'] and live['controller_sources']==state['controller_sources']
        assert live['server_command']==state['server_command']==profile.official_command()
        assert profile.normalized_environment(live['profile_environment'],official)==profile.normalized_environment(state['profile_environment'],root)
        if (official/'observed_server_environment.json').exists():
            observed=read(official/'observed_server_environment.json');assert observed['environment']==live['profile_environment'] and observed['command']==live['server_command'];official_verified=True
    tokens=[r['usage']['output_tokens'] for r in rows]
    archive=dict(complete=True,independently_rechecked=True,fresh_Math=True,correct=quality['correct'],total=105,accuracy=quality['accuracy'],strict_over95_passed=quality['accuracy']>.95,
        generation_stop_reasons=dict(Counter(r['stop_reason'] for r in rows)),output_tokens=dict(total=sum(tokens),minimum=min(tokens),median=statistics.median(tokens),maximum=max(tokens)),
        runtime_source_count=359,environment_count=62,separate_controller_source_count=len(state['controller_sources']),compile_cache_disabled=True,
        actual_C8_all_42_GateUp_integer_path_verified=True,actual_C8_LM_BF16_path_verified=True,LM_M64_integer_not_exercised_by_C8_Math=True,actual_new_prefill_ready_path_all_42_GateUp_verified=True,Math_actual_new_prefill_M=sorted({r['M'] for r in actual}),
        all_43_banks_match_successful_screen_and_previous_qualified_Math=True,extra_packed_weight_bytes=1326938112,
        full_responses_retained_in_container_only=True,per_question_archive_has_scores_usage_hashes_only=True,
        official_same_profile_and_actual_environment_verified=official_verified,official_phase_at_archive=phase,official_not_claimed_complete=phase!='complete',
        raw_root=str(root),official_root=str(official),raw_files=quality['raw_files'],verifier_sha256=sha(__file__))
    runtime_compact=dict(complete=True,actual_new_prefill_ready_prefixes=sorted({r['prefix'] for r in actual}),actual_new_prefill_ready_M=sorted({r['M'] for r in actual}),
        before_capture_warm_prefixes=route['warmup_GateUp_M2048_prefixes'],native_capture=route['native_capture'],native_linear=route['native_linear'],
        actual_C8_GateUp_integer_capture_and_replay_verified=True,actual_C8_LM_BF16_path_verified=True,actual_LM_head_outside_model_Graph_calls_verified=runtime['actual_LM_head_outside_model_Graph_calls_verified'],
        all_43_banks_match_previous_Math=True,actual_KV_blocks=runtime['actual_KV_blocks'],zero_preemptions=runtime['zero_preemptions'],
        future_reservations_released=runtime['future_reservation']['outstanding_blocks']==0,server_command=state['server_command'],profile_environment=env,
        frozen_sources=state['frozen_sources'],controller_sources=state['controller_sources'],raw_runtime_audit_in_accuracy_file=True)
    write('PER_QUESTION_32g.json',dict(total=105,correct=quality['correct'],rows=rows));write('ACCURACY_GATE_32g.json',{k:v for k,v in quality.items() if k not in ('profile','frozen_sources','profile_environment','controller_sources')})
    write('RUNTIME_BINDING_GATE_32g.json',runtime_compact);write('PACKED_WEIGHT_SIGNATURE_32g.json',banks);write('QUALITY_ARCHIVE_32g.json',archive)
    report=read(run/'reports/minicpm/math_500.json');write('EVALSCOPE_REPORT_32g.json',{k:report[k] for k in ('name','num','dataset_name','model_name','execution_summary','metrics') if k in report})
    print(json.dumps(archive))
if __name__=='__main__':main()
