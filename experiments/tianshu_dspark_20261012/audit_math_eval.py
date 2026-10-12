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
    proposal_count=spec['num_speculative_tokens']
    assert proposal_count in (1,2,3,7) and spec['draft_sample_method']=='probabilistic'
    assert state['environment_flags']['TIANSHU_DSPARK_COMPAT']=='1'
    routes=state['adapter_records']
    assert any(x['probabilistic'] and x['markov_head_used'] and x['block_size']==proposal_count for x in routes)
    assert all(x.get('backbone_block_size',7)==7 and x['query_slots_per_request']==7 for x in routes)
    if proposal_count<7:
        lookahead=[json.loads(p.read_text()) for p in (root/'adapter_records/scheduler_lookahead').glob('*.json')]
        assert lookahead and all(x['target_proposals']==proposal_count and x['lookahead_after']>=7 for x in lookahead)
        capacities=[json.loads(p.read_text()) for p in (root/'adapter_records/query_buffers').glob('*.json')]
        assert capacities and all(x['positions_capacity']>=x['max_batch_size']*7 and
            x['slot_mapping_capacity']>=x['max_batch_size']*7 and
            x['arange_capacity']>=x['max_context_tokens']+x['max_batch_size']*7+1 for x in capacities)
    threshold_compat=state['environment_flags'].get('TIANSHU_DSPARK_FP32_THRESHOLDS')=='1'
    assert all(x['target_rejection_sampler_modified']==threshold_compat for x in routes)
    if threshold_compat:
        assert all(x['positive_fp32_rejection_thresholds'] and not x['target_rejection_kernel_modified'] for x in routes)
        threshold_routes=[json.loads(p.read_text()) for p in (root/'adapter_records/rejection_threshold').glob('*.json')]
        assert threshold_routes and all(x['dtype']=='torch.float32' and 0<x['minimum']<=x['maximum']<=1.
            and not x['rejection_kernel_modified'] and not x['probability_ratio_changed'] for x in threshold_routes)
    else:threshold_routes=[]
    index_routes=[json.loads(p.read_text()) for p in (root/'adapter_records/fixed_index').glob('*.json')]
    if state['environment_flags'].get('TIANSHU_DSPARK_FIXED_INDEX')=='1':
        assert index_routes and all(x['fixed_launch'] for x in index_routes)
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
        'fixed_index_routes':index_routes,
        'rejection_threshold_routes':threshold_routes,
        'evaluation_command':command,'gpu_idle_after':True,
        'raw_files':{str(p.relative_to(root)):{'bytes':p.stat().st_size,
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (report_path,review_path,prediction_path)}}
    (root/'accuracy_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
