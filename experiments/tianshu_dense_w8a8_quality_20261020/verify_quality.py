# SPDX-License-Identifier: Apache-2.0
"""Independent new quality archive: scores only; exact 43-bank provenance."""
from collections import Counter
import hashlib,json,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_dense_w8a8_validation_20261020'))
import framework_profile as profile

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):(BASE/name).write_text(json.dumps(value,indent=2)+'\n')
def bank_signature(runtime):
    bank=runtime['dense_int8_record'];assert bank['load_complete'] and bank['actual_packed_banks']==43
    return sorted([{key:r[key] for key in ('role','prefix','shape','original_BF16_sha256','packed_sha256','scales_sha256','packed_bytes')} for r in bank['banks']],key=lambda r:r['prefix'])

def main():
    root=Path(sys.argv[1]);official=Path(sys.argv[2]);state=read(root/'result.json')
    assert state['phase']=='complete' and state['stage']=='math'
    profile.frozen_check(state['frozen_sources']);profile.frozen_check(state['controller_sources'])
    assert len(state['frozen_sources'])==311 and len(state['controller_sources'])==9 and len(state['profile_environment'])==57
    assert state['profile_environment']['VLLM_DISABLE_COMPILE_CACHE']=='1'
    runtime=profile.audit_runtime(root,state,official=False)
    assert runtime['complete'] and runtime['verified_dense_batch']==runtime['verified_batch']==8
    assert runtime['actual_42_GateUp_banks_capture_and_replay_verified'] and runtime['actual_LM_head_outside_model_Graph_calls_verified']
    quality=profile.audit_math(root,state,runtime,write=False);assert quality==read(root/'accuracy_audit.json')
    assert quality['complete'] and quality['total']==105 and quality['accuracy']>.95 and quality['actual_C8_dense_integer_path_verified']
    screened=read(Path(state['screen_root'])/'candidate/result.json');signature=bank_signature(runtime)
    assert signature==bank_signature(screened['profile_audit'])
    component=read(Path('/workspace/benchmark-results/20261020_dense_w8a8_GPU_tianshu/result.json'))
    for bank in signature:
        if bank['prefix']=='model.layers.0.mlp.gate_up_proj' or bank['role']=='lm_head':
            tested=component['weights'][bank['role']]
            assert bank['original_BF16_sha256']==tested['BF16_bits_sha256'] and bank['packed_sha256']==tested['packed_sha256'] and bank['scales_sha256']==tested['scales_sha256']
    run=next(root.glob('evalscope/*/reports/minicpm/math_500.json')).parents[2]
    predictions=run/'predictions/minicpm/math_500_Level 3.jsonl';reviews=run/'reviews/minicpm/math_500_Level 3.jsonl'
    pbytes=predictions.read_bytes().splitlines();rbytes=reviews.read_bytes().splitlines()
    pd={json.loads(line)['index']:(json.loads(line),line) for line in pbytes};rd={json.loads(line)['index']:(json.loads(line),line) for line in rbytes}
    assert len(pbytes)==len(rbytes)==len(pd)==len(rd)==105 and set(pd)==set(rd)==set(range(105))
    rows=[]
    for index in range(105):
        p,pl=pd[index];r,rl=rd[index];out=p['model_output'];score=r['sample_score']['score']
        assert out['error'] is None and len(out['choices'])==1 and score['status']=='success'
        accuracy=score['value']['accuracy'];assert accuracy in (0,1)
        usage=out['usage'];assert usage['input_tokens']>0 and 0<usage['output_tokens']<=32768
        assert usage['input_tokens']+usage['output_tokens']==usage['total_tokens']
        rows.append(dict(index=index,accuracy=accuracy,stop_reason=out['choices'][0]['stop_reason'],usage=usage,
            original_prediction_line_SHA256_without_newline=hashlib.sha256(pl).hexdigest(),original_review_line_SHA256_without_newline=hashlib.sha256(rl).hexdigest()))
    assert sum(r['accuracy'] for r in rows)==quality['correct']
    for relative,metadata in quality['raw_files'].items():assert sha(root/relative)==metadata['sha256'] and (root/relative).stat().st_size==metadata['bytes']
    official_verified=False;phase=None
    if (official/'observed_server_environment.json').exists():
        live=read(official/'state.json');phase=live['phase']
        assert live['stage']=='official' and live['frozen_sources']==state['frozen_sources'] and live['controller_sources']==state['controller_sources']
        assert live['server_command']==state['server_command']==profile.official_command()
        assert profile.normalized_environment(live['profile_environment'],official)==profile.normalized_environment(state['profile_environment'],root)
        observed=read(official/'observed_server_environment.json');assert observed['environment']==live['profile_environment'] and observed['command']==live['server_command']
        official_verified=True
    tokens=[r['usage']['output_tokens'] for r in rows]
    archive=dict(complete=True,independently_rechecked=True,fresh_Math=True,correct=quality['correct'],total=105,accuracy=quality['accuracy'],strict_over95_passed=True,
        generation_stop_reasons=dict(Counter(r['stop_reason'] for r in rows)),output_tokens=dict(total=sum(tokens),minimum=min(tokens),median=statistics.median(tokens),maximum=max(tokens)),
        runtime_source_count=311,environment_count=57,separate_controller_source_count=9,compile_cache_disabled=True,
        actual_C8_all_42_GateUp_and_LM_head_integer_path_verified=True,all_43_banks_match_the_successful_screen=True,actual_layer0_and_LM_weight_quantization_matches_component=True,
        extra_packed_weight_bytes=1326938112,full_responses_retained_in_container_only=True,per_question_archive_has_scores_usage_hashes_only=True,
        official_same_profile_and_actual_environment_verified=official_verified,official_phase_at_archive=phase,official_not_claimed_complete=phase!='complete',
        raw_root=str(root),official_root=str(official),raw_files=quality['raw_files'],verifier_sha256=sha(__file__))
    write('PER_QUESTION_32g.json',dict(total=105,correct=quality['correct'],rows=rows));write('ACCURACY_GATE_32g.json',quality)
    write('RUNTIME_BINDING_GATE_32g.json',runtime);write('PACKED_WEIGHT_SIGNATURE_32g.json',signature);write('QUALITY_ARCHIVE_32g.json',archive)
    report=read(run/'reports/minicpm/math_500.json');write('EVALSCOPE_REPORT_32g.json',{key:report[key] for key in ('name','num','dataset_name','model_name','execution_summary','metrics') if key in report})
    print(json.dumps(archive))
if __name__=='__main__':main()
