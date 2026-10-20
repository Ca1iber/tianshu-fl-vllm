# SPDX-License-Identifier: Apache-2.0
"""Recheck fresh Math, actual bindings and matching live official provenance.

Archive scores/usage/hashes only. Keep prompts and full responses in container.
"""
from collections import Counter
import hashlib,json,statistics,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_future_kv_validation_20261020'))
import framework_profile as profile

def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,obj):(BASE/name).write_text(json.dumps(obj,indent=2)+'\n')

def main():
    root=Path(sys.argv[1]);official=Path(sys.argv[2])
    state=read(root/'result.json');assert state['phase']=='complete' and state['stage']=='math'
    quality=profile.audit_math(root,state,state['profile_environment'],write=False)
    assert quality==read(root/'accuracy_audit.json')
    assert quality['complete'] and quality['total']==105 and quality['accuracy']>.95
    profile.frozen_check(state['frozen_sources']);profile.frozen_check(state['controller_sources'])
    runtime=profile.audit_runtime(root,state,official=False)
    assert runtime['complete'] and runtime['metadata_fusion_excluded']
    assert len(state['frozen_sources'])==261 and len(state['controller_sources'])==7
    run=next(root.glob('evalscope/*/reports/minicpm/math_500.json')).parents[2]
    prediction=run/'predictions/minicpm/math_500_Level 3.jsonl'
    reviews=run/'reviews/minicpm/math_500_Level 3.jsonl'
    pbytes=prediction.read_bytes().splitlines();rbytes=reviews.read_bytes().splitlines()
    pd={json.loads(line)['index']:(json.loads(line),line) for line in pbytes}
    rd={json.loads(line)['index']:(json.loads(line),line) for line in rbytes}
    assert len(pbytes)==len(rbytes)==len(pd)==len(rd)==105
    assert set(pd)==set(rd)==set(range(105))
    rows=[]
    for index in range(105):
        p,pl=pd[index];r,rl=rd[index];output=p['model_output'];score=r['sample_score']['score']
        assert output['error'] is None and len(output['choices'])==1 and score['status']=='success'
        value=score['value']['accuracy'];assert value in (0,1)
        usage=output['usage'];assert 0<usage['output_tokens']<=32768 and usage['input_tokens']>0
        assert usage['input_tokens']+usage['output_tokens']==usage['total_tokens']
        rows.append(dict(index=index,accuracy=value,stop_reason=output['choices'][0]['stop_reason'],usage=usage,
            original_prediction_line_SHA256_without_newline=hashlib.sha256(pl).hexdigest(),
            original_review_line_SHA256_without_newline=hashlib.sha256(rl).hexdigest()))
    assert sum(row['accuracy'] for row in rows)==quality['correct']==103
    for relative,metadata in quality['raw_files'].items():
        assert digest(root/relative)==metadata['sha256'] and (root/relative).stat().st_size==metadata['bytes']
    live=read(official/'state.json')
    assert live['stage']=='official' and live['phase'] in ('starting_server','benchmarking','benchmark_finished_pending_audit','complete')
    assert live['frozen_sources']==state['frozen_sources'] and live['controller_sources']==state['controller_sources']
    assert live['server_command']==state['server_command']==profile.official_command()
    assert profile.normalized_environment(live['profile_environment'],official)==profile.normalized_environment(state['profile_environment'],root)
    observed=read(official/'observed_server_environment.json')
    assert observed['command']==live['server_command'] and observed['environment']==live['profile_environment']
    tokens=[r['usage']['output_tokens'] for r in rows]
    archive=dict(complete=True,independently_rechecked=True,fresh_Math=True,correct=103,total=105,accuracy=103/105,
        strict_over95_passed=True,generation_stop_reasons=dict(Counter(r['stop_reason'] for r in rows)),
        output_tokens=dict(total=sum(tokens),minimum=min(tokens),median=statistics.median(tokens),maximum=max(tokens)),
        runtime_source_count=261,environment_count=52,separate_controller_source_count=7,
        official_ongoing_same_profile_and_actual_environment_verified=True,official_not_yet_claimed_complete=live['phase']!='complete',
        raw_root=str(root),official_root=str(official),raw_files=quality['raw_files'],
        per_question_archive_contains_scores_usage_and_hashes_only=True,full_responses_retained_in_container_only=True,
        verifier_sha256=digest(__file__))
    write('PER_QUESTION_32g.json',dict(total=105,correct=103,rows=rows))
    write('RUNTIME_BINDING_GATE_32g.json',runtime)
    write('QUALITY_ARCHIVE_32g.json',archive)
    print(json.dumps(archive,ensure_ascii=False))
if __name__=='__main__':main()
