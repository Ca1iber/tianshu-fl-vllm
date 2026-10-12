# SPDX-License-Identifier: Apache-2.0
"""Audit all105 independent Level3 predictions and actual DSpark use."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve()
    state=json.loads((root/'result.json').read_text())
    assert state['phase']=='evaluation_finished_pending_report_audit'
    assert state['accuracy_test'] and not state['official_benchmark']
    assert state['evalscope_exit_code']==0 and state['gpu_idle_after']
    spec=state['speculative_config']
    assert spec['num_speculative_tokens']==7 and spec['draft_sample_method']=='probabilistic'
    assert state['environment_flags']['TIANSHU_DSPARK_COMPAT']=='1'
    routes=state['adapter_records']
    assert any(x['probabilistic'] and x['markov_head_used'] and x['block_size']==7 for x in routes)
    assert all(not x['target_rejection_sampler_modified'] for x in routes)
    command=state['evaluation_command']
    assert json.loads(command[command.index('--generation-config')+1])=={
        'temperature':1.,'top_p':.95,'max_tokens':32768}
    assert command[command.index('--eval-batch-size')+1]=='8'
    assert json.loads(command[command.index('--dataset-args')+1])=={
        'math_500':{'dataset_id':'/workspace/evalscope-datasets/math_500','subset_list':['Level 3']}}
    reports=list(root.glob('evalscope/*/reports/minicpm/math_500.json'));assert len(reports)==1
    report_path=reports[0];report=json.loads(report_path.read_text());execution=report['execution_summary']
    assert execution['requested']==execution['succeeded']==report['num']==105
    assert execution['errored']==0 and execution['incomplete'] is False
    metric=next(x for x in report['metrics'] if x['identity']['name']=='accuracy');assert metric['num']==105
    run=report_path.parents[2]
    review_path=run/'reviews/minicpm/math_500_Level 3.jsonl'
    prediction_path=run/'predictions/minicpm/math_500_Level 3.jsonl'
    reviews=[json.loads(x) for x in review_path.read_text().splitlines()]
    predictions=[json.loads(x) for x in prediction_path.read_text().splitlines()]
    assert len(reviews)==len(predictions)==105
    assert len({x['index'] for x in reviews})==len({x['index'] for x in predictions})==105
    assert {x['index'] for x in reviews}=={x['index'] for x in predictions}
    assert all(x['model_output']['error'] is None for x in predictions)
    scores=[x['sample_score']['score'] for x in reviews]
    assert all(x['status']=='success' for x in scores)
    values=[x['value']['accuracy'] for x in scores];assert all(x in (0,1) for x in values)
    correct=int(sum(values));assert abs(metric['score']-correct/105)<.000051
    stop=Counter(x['model_output']['choices'][0]['stop_reason'] for x in predictions)
    audit={'complete':True,'accuracy_test':True,'official_benchmark':False,'reused_accuracy':False,
        'audited_utc':datetime.now(timezone.utc).isoformat(),'correct':correct,'total':105,
        'accuracy':correct/105,'report_accuracy':metric['score'],'official_requirement':'accuracy >0.95',
        'official_requirement_passed':correct/105>.95,'execution_summary':execution,
        'incorrect_indices':[x['index'] for x,v in zip(reviews,values) if not v],
        'generation_stop_reasons':dict(stop),'server_command':state['command'],
        'speculative_config':spec,'environment_flags':state['environment_flags'],
        'installed_source_sha256':state['installed_source_sha256'],'adapter_records':routes,
        'evaluation_command':command,'gpu_idle_after':True,
        'raw_files':{str(p.relative_to(root)):{'bytes':p.stat().st_size,
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (report_path,review_path,prediction_path)}}
    (root/'accuracy_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
