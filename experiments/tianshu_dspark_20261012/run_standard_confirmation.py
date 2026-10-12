# SPDX-License-Identifier: Apache-2.0
"""Compare the selected Spec profile with the verified best using1K output."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    base=Path(__file__).resolve().parents[1]
    source=Path('/workspace/benchmark-results/20261012_speculative_dspark_tianshu')
    lease=Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    deadline=datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])
    state={'phase':'waiting','runner_pid':os.getpid(),'official_benchmark':False,
           'accuracy_test_started':False,'jobs':[]}
    def save():
        p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')
    def remaining():return (deadline-datetime.now(timezone.utc)).total_seconds()
    def run(name,command):
        assert remaining()>0
        state.update(phase='running',current_job=name);save()
        with (root/(name+'_runner.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd='/workspace/vllm-plugin-FL')
        state['jobs'].append({'name':name,'exit_code':result.returncode});save()
        assert result.returncode==0,name+' failed'
    save()
    try:
        prerequisite=source/'matched_warmup_controller_state.json'
        while not prerequisite.exists() or json.loads(prerequisite.read_text())['phase']=='running':
            assert remaining()>0
            time.sleep(5)
        options=[]
        for name,flags in [('quick_graph_dspark_both_tiles_c64',[]),
                           ('quick_graph_dspark_matched_warmup_c64',['--matched-warmup'])]:
            p=source/name/'result.json'
            if not p.exists():continue
            d=json.loads(p.read_text());rows=d.get('performance_rounds',[])
            if d['phase']=='complete' and len(rows)==3 and all(not x['tuning_records'] for x in rows[1:]):
                options.append((statistics.mean(x['total_tokens_per_second'] for x in rows[1:]),name,flags))
        assert options,'No clean completed composite Spec profile'
        _,name,flags=max(options)
        state['selected_short_profile']=name
        candidate_flags=['--zero-verify','--verify-tile','--prefill-tile',
                         '--target-graphs','--concurrency','64']+flags
        state['candidate_flags']=candidate_flags;save()
        # Budget time for both complete three-round profiles before starting.
        assert remaining()>3600,'Less than1h remains for standard confirmation'
        control=root/'current_best_control';candidate=root/'selected_spec'
        run('current_best_control',['python3',str(base/'tianshu_native_prefill_20261012/run_service_trial.py'),
            '--output',str(control),'--buckets','0','--gather','0','--hybrid','1','--adaptive','1',
            '--max-batched-tokens','2048','--tile','1','--native-prefill','1',
            '--output-tokens','1024','--concurrency','64','--requests','128','--rounds','3'])
        run('selected_spec',['python3',str(base/'tianshu_dspark_20261012/run_smoke.py'),
            '--output',str(candidate),'--quick-performance','--output-tokens','1024',
            '--seed-base','2026101200','--seed-round-stride','128']+candidate_flags)
        a=json.loads((control/'result.json').read_text());b=json.loads((candidate/'result.json').read_text())
        assert a['phase']==b['phase']=='complete'
        rows_a=a['short_4k_rounds'];rows_b=b['performance_rounds'];assert len(rows_a)==len(rows_b)==3
        assert all(x['successful_requests']==128 for x in rows_a+rows_b)
        assert all(x['usage']['prompt_tokens']==4096 and x['usage']['completion_tokens']==1024
                   for row in rows_a for x in row['responses'])
        assert all(x['response']['usage']['prompt_tokens']==4096 and x['response']['usage']['completion_tokens']==1024
                   for row in rows_b for x in row['responses'])
        for round_index,row in enumerate(rows_a):
            assert all(x['seed']==2026101200+128*round_index+i for i,x in enumerate(row['responses']))
        assert b['workload']['seed_base']==2026101200 and b['workload']['seed_round_stride']==128
        av=[x['total_tokens_per_second'] for x in rows_a[1:]];bv=[x['total_tokens_per_second'] for x in rows_b[1:]]
        ma,mb=statistics.mean(av),statistics.mean(bv)
        clean=all(not x['tuning_records'] for x in rows_a[1:]+rows_b[1:])
        gain=100*(mb/ma-1)
        result={'official_benchmark':False,'accuracy_test':False,'control_profile':'verified best native Prefill+adaptive',
            'candidate_profile':name,'workload':{'input_tokens':4096,'output_tokens':1024,
                'concurrency':64,'requests_per_round':128,'rounds':3,'excluded_rounds':[1]},
            'same_input_seed_recipe':True,'control_mean_total_tok_s':ma,'candidate_mean_total_tok_s':mb,
            'throughput_change_percent':gain,'measured_tuning_clean':clean,
            'control_measured_rows':av,'candidate_measured_rows':bv,
            'control_mean_ttft_ms':1000*statistics.mean(x['mean_ttft_seconds'] for x in rows_a[1:]),
            'candidate_mean_ttft_ms':1000*statistics.mean(x['mean_ttft_seconds'] for x in rows_b[1:]),
            'candidate_flags':candidate_flags,'promising':clean and gain>=3 and min(bv)>max(av)}
        (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['promising'] and remaining()>7200:
            quality=root/'math';state['accuracy_test_started']=True;save()
            run('math',['python3',str(base/'tianshu_dspark_20261012/run_smoke.py'),
                        '--output',str(quality),'--math-level3']+candidate_flags)
            run('math_audit',['python3',str(base/'tianshu_dspark_20261012/audit_math_eval.py'),
                              '--output',str(quality)])
            audit=json.loads((quality/'accuracy_audit.json').read_text())
            state['quality_requirement_passed']=audit['official_requirement_passed']
            state['quality_correct']=audit['correct'];state['quality_total']=audit['total']
        state['phase']='complete'
    except Exception as error:
        state.update(phase='failed',error=str(error))
    save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
