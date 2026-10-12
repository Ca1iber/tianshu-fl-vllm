# SPDX-License-Identifier: Apache-2.0
"""Audit complete unchanged4K protocol plus matching independent Spec quality."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quality-results',type=Path,required=True)
    parser.add_argument('--previous-total-tok-s',type=float,default=2712.55)
    parser.add_argument('--previous-ttft-ms',type=float,default=4720.38)
    args=parser.parse_args();root=args.output.resolve()
    state=json.loads((root/'result.json').read_text())
    quality=json.loads((args.quality_results/'accuracy_audit.json').read_text())
    assert state['phase']=='complete' and state['gpu_idle_after'] and state['official_benchmark']
    assert state['benchmark_exit_code']==0 and state['runs']==state['completed_runs']==4 and state['skip_first']==1
    assert quality['complete'] and quality['total']==105 and not quality['reused_accuracy']
    assert quality['accuracy']>.95 and quality['correct']/105>.95 and quality['official_requirement_passed']
    assert state['environment_flags'].get('TIANSHU_DSPARK_FP32_THRESHOLDS')=='1'
    assert quality['installed_source_sha256']==state['installed_source_sha256']
    assert state['independent_math']=={'correct':quality['correct'],'total':105,'accuracy':quality['accuracy']}
    keys=('TIANSHU_DSPARK_COMPAT','TIANSHU_DSPARK_ZERO_VERIFY','TIANSHU_DSPARK_SAMPLE_VIEW',
          'TIANSHU_DSPARK_VERIFY_TILE','TIANSHU_DSPARK_PREFILL_TILE','TIANSHU_DSPARK_MATCHED_WARMUP',
          'TIANSHU_DSPARK_FIXED_INDEX','TIANSHU_DSPARK_FP32_THRESHOLDS')
    assert all(state['environment_flags'].get(k,'0')==quality['environment_flags'].get(k,'0') for k in keys)
    benchmark=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
    assert (hashlib.sha256(benchmark.read_bytes()).hexdigest()==state['official_script_sha256']==
            '0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565')
    rows=state['raw_runs'];assert len(rows)==4
    for row in rows:
        assert row['Run Status']=='SUCCESS' and int(float(row['Successful Requests']))==256
        assert [int(row[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
        assert int(float(row['Total Input Tokens']))==1048576 and int(float(row['Total Output Tokens']))==262144
    summary=state['official_summary']
    for key in ('Total tok/s','Output tok/s','Mean TTFT (ms)'):
        assert abs(float(summary[key])-statistics.mean(float(x[key]) for x in rows[1:]))<.011
    rates=[float(x['Total tok/s']) for x in rows[1:]]
    total=float(summary['Total tok/s']);ttft=float(summary['Mean TTFT (ms)'])
    proposal_count=state['speculative_config']['num_speculative_tokens']
    result={'complete':True,'official_benchmark':True,'audited_utc':datetime.now(timezone.utc).isoformat(),
        'profile':f'DSpark{proposal_count}-step probability draft +target native L0/GQA verification and Prefill',
        'total_tok_s':total,'output_tok_s':float(summary['Output tok/s']),'ttft_ms':ttft,
        'math_correct':quality['correct'],'math_total':105,'math_accuracy':quality['accuracy'],
        'requests_successful':1024,'excluded_runs':[1],
        'previous_best_total_tok_s':args.previous_total_tok_s,'previous_best_ttft_ms':args.previous_ttft_ms,
        'vs_previous_best_throughput_percent':100*(total/args.previous_total_tok_s-1),
        'vs_previous_best_ttft_percent':100*(ttft/args.previous_ttft_ms-1),
        'vs_public_throughput_percent':100*(total/2028.01-1),
        'vs_public_ttft_percent':100*(ttft/11573.53-1),
        'postfirst_spread_percent':100*(max(rates)-min(rates))/statistics.mean(rates),
        'benchmark_runtime_tuning_record_count':len(state['benchmark_runtime_tuning_records']),
        'environment_flags':state['environment_flags'],'server_command':state['command'],
        'benchmark_command':state['benchmark_command'],'installed_source_sha256':state['installed_source_sha256'],
        'independent_quality_reused':False,'optimized_full_16k_not_run':True,
        'adopt_as_best':total>args.previous_total_tok_s}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# DSpark 正式4K结果','',
        f"原始脚本四轮排首轮，均值{total:.2f}total /{result['output_tok_s']:.2f}output tok/s、TTFT{ttft:.2f}ms。独立Math{quality['correct']}/105，准确率{quality['accuracy']*100:.6f}%，严格>95%。",'',
        '|轮次|total tok/s|TTFT/ms|duration/s|纳入均值|','|---|---:|---:|---:|---|']
    lines += [f"|{i+1}|{x['Total tok/s']}|{x['Mean TTFT (ms)']}|{x['Benchmark Duration (s)']}|{'否' if i==0 else '是'}|" for i,x in enumerate(rows)]
    lines += ['',f"相对前最佳{args.previous_total_tok_s:.2f}，吞吐{result['vs_previous_best_throughput_percent']:+.6f}%；后3轮极差/均值{result['postfirst_spread_percent']:.6f}%。",
              '',f"计时阶段共{result['benchmark_runtime_tuning_record_count']}条调优记录。若非零，不从全窗计数推断仅首轮调优。",
              '', f'同独立质量评测的冻结模型钩子和运行参数，{proposal_count}-step Markov概率draft、7槽预训练骨干，正FP32随机拒绝阈值兼容；原始概率比、拒绝kernel及恢复采样保留。不修改目标模型权重、数据集、原始benchmark脚本。未运行优化版完整16K。']
    (root/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
