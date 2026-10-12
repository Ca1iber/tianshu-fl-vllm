# SPDX-License-Identifier: Apache-2.0
"""Audit only complete official runs and their matching independent quality."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--math-results',type=Path,required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    state = json.loads((root/'result.json').read_text())
    quality = json.loads((args.math_results/'accuracy_audit.json').read_text())
    assert state['phase']=='complete' and state['gpu_idle_after'] and state['benchmark_exit_code']==0
    assert state['completed_runs']==state['runs']==4 and state['skip_first']==1
    assert quality['complete'] and quality['total']==105 and quality['accuracy']>.95
    assert not quality['reused_original_accuracy'] and quality['experimental_candidate']=='native-prefill'
    assert state['candidate_hook']==quality['candidate_hook']
    assert state['experimental_kernel']==quality['experimental_kernel']
    rows=state['raw_runs'];assert len(rows)==4
    for row in rows:
        assert row['Run Status']=='SUCCESS' and int(float(row['Successful Requests']))==256
        assert [int(row[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
        assert int(float(row['Total Input Tokens']))==1048576 and int(float(row['Total Output Tokens']))==262144
    summary=state['official_summary']
    for key in ('Total tok/s','Output tok/s','Mean TTFT (ms)'):
        assert abs(float(summary[key])-statistics.mean(float(x[key]) for x in rows[1:]))<.011
    benchmark=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
    assert hashlib.sha256(benchmark.read_bytes()).hexdigest()==state['official_script_sha256']=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
    generated={x['generated_kernel_sha256'] for x in quality['candidate_routing_records']}
    assert state['candidate_routing_records'] and all(x['generated_kernel_sha256'] in generated for x in state['candidate_routing_records'])
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in (args.math_results/'source_snapshot').glob('*.py'):
        (snapshot/path.name).write_bytes(path.read_bytes())
    (snapshot/'benchmark_throughput_serve.py').write_bytes(benchmark.read_bytes())
    total=float(summary['Total tok/s']);ttft=float(summary['Mean TTFT (ms)'])
    values=[float(x['Total tok/s']) for x in rows[1:]]
    result={'complete':True,'official_benchmark':True,'audited_utc':datetime.now(timezone.utc).isoformat(),
        'total_tok_s':total,'output_tok_s':float(summary['Output tok/s']),'ttft_ms':ttft,
        'math_correct':quality['correct'],'math_total':105,'math_accuracy':quality['accuracy'],
        'requests_successful':1024,'excluded_runs':[1],
        'vs_previous_best_throughput_percent':100*(total/2504.08-1),
        'vs_previous_best_ttft_percent':100*(ttft/6326.71-1),
        'vs_public_throughput_percent':100*(total/2028.01-1),
        'vs_public_ttft_percent':100*(ttft/11573.53-1),
        'vs_original_measured_throughput_percent':100*(total/1876.67-1),
        'postfirst_spread_percent':100*(max(values)-min(values))/statistics.mean(values),
        'adopt_as_best':total>2504.08,'profile':'Hybrid routing + adaptive Decode + native GQA Prefill M256/KV128/w16/s1, no-sink L0, budget2048',
        'candidate_hook':quality['candidate_hook'],'experimental_kernel':quality['experimental_kernel'],
        'optimized_full_16k_not_run':True}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 天数原生 GQA 预填充正式4K结果','',
        f"完整四轮排除首轮，均值{total:.2f}total tok/s、{result['output_tok_s']:.2f}output tok/s、TTFT{ttft:.2f}ms。对应独立Math{quality['correct']}/105（{quality['accuracy']*100:.6f}%），严格>95%。四轮均256/256成功，总计1024请求。",'',
        '|轮次|总tok/s|TTFT/ms|duration/s|纳入均值|','|---|---:|---:|---:|---|']
    lines.extend(f"|{i+1}|{x['Total tok/s']}|{x['Mean TTFT (ms)']}|{x['Benchmark Duration (s)']}|{'否' if i==0 else '是'}|" for i,x in enumerate(rows))
    lines+=['',f"相对先前正式最佳2504.08tok/s，吞吐{result['vs_previous_best_throughput_percent']:+.6f}%；相对公开2028.01tok/s，吞吐{result['vs_public_throughput_percent']:+.6f}%。后3轮极差/均值{result['postfirst_spread_percent']:.6f}%。",'',
        '安装源码未改；进程钩子复用原生attention，只在支持的预填充路径调整GQA tile、网格和无sink在线softmax初始分母。Decode与混合批次Decode prefix分段/算术保持原配置。未运行优化版完整16K，不将短测或算子耗时当作官方分数。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
