# SPDX-License-Identifier: Apache-2.0
"""Serialize short-KV oracle checks and the selected Decode service candidate."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dependency',type=Path,required=True);parser.add_argument('--dependency-pid',type=int,required=True)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--numerical-results',type=Path)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    base=Path(__file__).resolve().parents[1]
    deadline=datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    state={'phase':'waiting','runner_pid':os.getpid(),'jobs':[],'official_benchmark':False}
    def save():
        p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')
    def run(name,command):
        assert datetime.now(timezone.utc)<deadline
        state.update(phase='running',current_job=name);save()
        with (root/(name+'_runner.log')).open('w') as log:
            p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd='/workspace/vllm-plugin-FL')
        state['jobs'].append({'name':name,'exit_code':p.returncode});save()
        assert p.returncode==0,name+' failed'
    save()
    try:
        while not (args.dependency/'result.json').exists():
            assert datetime.now(timezone.utc)<deadline
            process=Path(f'/proc/{args.dependency_pid}/status')
            if not process.exists() or 'State:\tZ' in process.read_text():
                state['dependency_ended_without_result']=True;save();break
            time.sleep(5)
        while 'No running processes found' not in subprocess.check_output(['ixsmi'],text=True):
            assert datetime.now(timezone.utc)<deadline
            time.sleep(5)
        numerical=root/'short_boundaries'
        if args.numerical_results:
            shutil.copytree(args.numerical_results,numerical)
            state['reused_completed_numerical_results']=str(args.numerical_results);save()
        else:
            run('short_boundaries',['python3',str(base/'tianshu_decode_launch_20261012/probe_launch.py'),
                '--short-boundaries','--selected-tile','16','--output',str(numerical)])
        numeric=json.loads((numerical/'result.json').read_text());assert numeric['complete'] and len(numeric['cases'])==14
        assert all(r['original_vs_fp32']['passed'] and r['baseline_tail_preserved'] and
            all(c['passed'] for c in r['cpu_reference_checks']) and len(r['variants'])==2 and
            r['variants'][0]['configuration']=={'TILE_SIZE':16,'num_warps':4,'num_stages':1} and
            r['variants'][0]['candidate_vs_fp32']['passed'] and r['variants'][0]['candidate_vs_native']['passed'] and
            r['variants'][0]['tails_preserved'] for r in numeric['cases'])
        candidate=root/'candidate'
        run('service',['python3',str(base/'tianshu_native_prefill_20261012/run_service_trial.py'),
            '--output',str(candidate),'--native-prefill','1','--tile','1','--hybrid','1','--adaptive','1',
            '--buckets','0','--gather','0','--max-batched-tokens','2048','--output-tokens','1024',
            '--concurrency','64','--requests','128','--rounds','3','--decode-launch','1'])
        a=json.loads((args.control/'result.json').read_text());b=json.loads((candidate/'result.json').read_text())
        assert a['phase']==b['phase']=='complete' and a['workload']==b['workload']
        assert a['installed_sources']==b['installed_sources'] and a['tile_hook']==b['tile_hook']
        assert len(b['short_4k_rounds'])==3 and all(r['successful_requests']==128 for r in b['short_4k_rounds'])
        assert all(x['usage']['prompt_tokens']==4096 and x['usage']['completion_tokens']==1024
                   for r in b['short_4k_rounds'] for x in r['responses'])
        assert all([x['seed'] for x in left['responses']]==[x['seed'] for x in right['responses']]
                   for left,right in zip(a['short_4k_rounds'],b['short_4k_rounds']))
        records=[json.loads(p.read_text()) for p in (candidate/'decode_launch_records').glob('*.json')]
        assert records and all(x['configuration']=={'TILE_SIZE':16,'num_warps':4,'num_stages':1} for x in records)
        av=[r['total_tokens_per_second'] for r in a['short_4k_rounds'][1:]]
        bv=[r['total_tokens_per_second'] for r in b['short_4k_rounds'][1:]]
        clean=all(not r['tuning_records'] for r in a['short_4k_rounds'][1:]+b['short_4k_rounds'][1:])
        gain=100*(statistics.mean(bv)/statistics.mean(av)-1)
        comparison={'official_benchmark':False,'independent_quality_pending':True,'control_reused':True,
            'control_source':str(args.control),'workload':b['workload'],'baseline_tok_s':statistics.mean(av),
            'candidate_tok_s':statistics.mean(bv),'throughput_gain_percent':gain,
            'baseline_ttft_ms':1000*statistics.mean(r['mean_ttft_seconds'] for r in a['short_4k_rounds'][1:]),
            'candidate_ttft_ms':1000*statistics.mean(r['mean_ttft_seconds'] for r in b['short_4k_rounds'][1:]),
            'timed_round_tuning_counts':[len(r['tuning_records']) for r in b['short_4k_rounds']],
            'measured_tuning_clean':clean,'candidate_measured_rows':bv,'baseline_measured_rows':av,
            'promising':clean and gain>=3 and min(bv)>max(av),'decode_records':records,'adopted_as_best':False}
        (root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
        state['phase']='complete'
    except Exception as error:state.update(phase='failed',error=str(error))
    save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
