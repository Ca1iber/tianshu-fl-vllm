# SPDX-License-Identifier: Apache-2.0
"""Audit the repaired1K-output lane, then gate independent Math and official4K."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    directory=Path(__file__).resolve().parent
    deadline=datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    flags=['--zero-verify','--verify-tile','--prefill-tile','--fixed-index','--fp32-thresholds',
           '--target-graphs','--concurrency','64','--request-timeout','3600']
    state={'phase':'waiting','runner_pid':os.getpid(),'candidate_flags':flags,
           'candidate_results':str(args.candidate),'control_results':str(args.control),
           'control_reused_with_same_profile':True,'accuracy_test_started':False,
           'official_benchmark_started':False,'jobs':[]}
    def remaining():return (deadline-datetime.now(timezone.utc)).total_seconds()
    def save():
        p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')
    def run(name,command):
        assert remaining()>0
        state.update(phase='running',current_job=name);save()
        with (root/(name+'_runner.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd='/workspace/vllm-plugin-FL')
        state['jobs'].append({'name':name,'exit_code':result.returncode});save()
        assert result.returncode==0,name+' failed'
    save()
    try:
        result_file=args.candidate/'result.json'
        while not result_file.exists():
            assert remaining()>0
            time.sleep(5)
        a=json.loads((args.control/'result.json').read_text());b=json.loads(result_file.read_text())
        assert a['phase']==b['phase']=='complete' and b['gpu_idle_after']
        rows_a=a['short_4k_rounds'];rows_b=b['performance_rounds']
        assert len(rows_a)==len(rows_b)==3
        assert all(x['successful_requests']==128 for x in rows_a+rows_b)
        assert all(x['usage']['prompt_tokens']==4096 and x['usage']['completion_tokens']==1024
                   for row in rows_a for x in row['responses'])
        assert all(x['response']['usage']['prompt_tokens']==4096 and x['response']['usage']['completion_tokens']==1024
                   for row in rows_b for x in row['responses'])
        for round_index,row in enumerate(rows_a):
            assert all(x['seed']==2026101200+128*round_index+i for i,x in enumerate(row['responses']))
        assert b['workload']['seed_base']==2026101200 and b['workload']['seed_round_stride']==128
        routes=list((args.candidate/'adapter_records/fixed_index').glob('*.json'))
        assert routes and all(json.loads(p.read_text())['fixed_launch'] for p in routes)
        # Ensure a later quality run uses exactly the measured kernel sources.
        frozen=Path(b['frozen_hook_directory']).parent
        core=('hooks/sitecustomize.py','fixed_index.py','rejection_compat.py','tianshu_dspark_adapter/model.py',
              'tianshu_dspark_adapter/proposer.py')
        for rel in core:assert (directory/rel).read_bytes()==(frozen/rel).read_bytes(),rel+' changed'
        source_sha={rel:hashlib.sha256((frozen/rel).read_bytes()).hexdigest() for rel in core}
        assert all(b['environment_flags'].get(k)=='1' for k in (
            'TIANSHU_DSPARK_COMPAT','TIANSHU_DSPARK_ZERO_VERIFY','TIANSHU_DSPARK_VERIFY_TILE',
            'TIANSHU_DSPARK_PREFILL_TILE','TIANSHU_DSPARK_FIXED_INDEX','TIANSHU_DSPARK_FP32_THRESHOLDS'))
        assert all(b['environment_flags'].get(k)=='0' for k in (
            'TIANSHU_DSPARK_SAMPLE_VIEW','TIANSHU_DSPARK_MATCHED_WARMUP'))
        av=[x['total_tokens_per_second'] for x in rows_a[1:]]
        bv=[x['total_tokens_per_second'] for x in rows_b[1:]]
        ma,mb=statistics.mean(av),statistics.mean(bv);gain=100*(mb/ma-1)
        clean=all(not x['tuning_records'] for x in rows_a[1:]+rows_b[1:])
        comparison={'official_benchmark':False,'accuracy_test':False,
            'control_profile':'verified native Prefill+Hybrid+adaptive, budget2048',
            'candidate_profile':'DSpark native L0+verify/Prefill tiles+dual-path fixed row index',
            'control_results':str(args.control),'candidate_results':str(args.candidate),
            'control_reused':True,'same_input_seed_recipe':True,
            'workload':{'input_tokens':4096,'output_tokens':1024,'concurrency':64,
                        'requests_per_round':128,'rounds':3,'excluded_rounds':[1]},
            'control_mean_total_tok_s':ma,'candidate_mean_total_tok_s':mb,
            'control_measured_rows':av,'candidate_measured_rows':bv,
            'throughput_change_percent':gain,'measured_tuning_clean':clean,
            'control_mean_ttft_ms':1000*statistics.mean(x['mean_ttft_seconds'] for x in rows_a[1:]),
            'candidate_mean_ttft_ms':1000*statistics.mean(x['mean_ttft_seconds'] for x in rows_b[1:]),
            'candidate_flags':flags,'candidate_source_sha256':source_sha,
            'promising':clean and gain>=3 and min(bv)>max(av)}
        (root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
        if comparison['promising'] and remaining()>7200:
            quality=root/'math';state['accuracy_test_started']=True;save()
            run('math',['python3',str(directory/'run_smoke.py'),'--output',str(quality),'--math-level3']+flags)
            run('math_audit',['python3',str(directory/'audit_math_eval.py'),'--output',str(quality)])
            audit=json.loads((quality/'accuracy_audit.json').read_text())
            state.update(quality_correct=audit['correct'],quality_total=105,
                         quality_requirement_passed=audit['official_requirement_passed'])
            if audit['official_requirement_passed'] and remaining()>3600:
                state['official_benchmark_started']=True;save()
                official=root/'official'
                run('official',['python3',str(directory/'run_smoke.py'),'--output',str(official),
                    '--official-4k','--quality-results',str(quality),'--standard-results',str(root)]+flags)
                run('official_audit',['python3',str(directory/'audit_official.py'),
                                     '--output',str(official),'--quality-results',str(quality)])
        state['phase']='complete'
    except Exception as error:
        state.update(phase='failed',error=str(error),error_type=type(error).__name__)
    save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
