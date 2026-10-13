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
    parser.add_argument('--previous-best-tok-s',type=float,default=915.15)
    parser.add_argument('--previous-best-ttft-ms',type=float,default=599262.03)
    args = parser.parse_args()
    root = args.output.resolve()
    state = json.loads((root/'result.json').read_text())
    quality = json.loads((args.math_results/'accuracy_audit.json').read_text())
    assert state['phase']=='complete' and state['gpu_idle_after'] and state['benchmark_exit_code']==0
    assert state['completed_runs']==state['runs']==4 and state['skip_first']==1
    assert quality['complete'] and quality['total']==105 and quality['accuracy']>.95
    assert not quality['reused_original_accuracy'] and quality['experimental_candidate']==state['experimental_candidate']
    assert state['candidate_hook']==quality['candidate_hook']
    assert state['experimental_kernel']==quality['experimental_kernel']
    assert state.get('buckets','0')==quality.get('buckets','0')
    assert state.get('gather','0')==quality.get('gather','0')
    assert state['max_batched_tokens']==quality['max_batched_tokens']
    if state.get('buckets')=='1':
        assert state['bucket_sources']==quality['bucket_sources']
        assert state['bucket_trace_hook']==quality['bucket_trace_hook']
        assert state['actual_gather_records'] and any(x['index_dtype']=='torch.int32' for x in state['actual_gather_records'])
        assert any(x['complete'] and len(x['calls'])==75 for x in state['bucket_warmup_records'])
    rows=state['raw_runs'];assert len(rows)==4
    for row in rows:
        assert row['Run Status']=='SUCCESS' and int(float(row['Successful Requests']))==128
        assert [int(row[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[16384,1024,64,128]
        assert int(float(row['Total Input Tokens']))==2097152 and int(float(row['Total Output Tokens']))==131072
    summary=state['official_summary']
    for key in ('Total tok/s','Output tok/s','Mean TTFT (ms)'):
        assert abs(float(summary[key])-statistics.mean(float(x[key]) for x in rows[1:]))<.011
    benchmark=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
    assert hashlib.sha256(benchmark.read_bytes()).hexdigest()==state['official_script_sha256']=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
    is_decode=state['experimental_candidate']=='decode-launch'
    quality_native=quality['native_prefill_routing_records'] if is_decode else quality['candidate_routing_records']
    official_native=state['native_prefill_routing_records'] if is_decode else state['candidate_routing_records']
    generated={x['generated_kernel_sha256'] for x in quality_native}
    assert official_native and all(x['generated_kernel_sha256'] in generated for x in official_native)
    if is_decode:
        assert state['candidate_routing_records'] and all(x['configuration']=={'TILE_SIZE':16,'num_warps':4,'num_stages':1}
            and x['kernel_sha256']==quality['experimental_kernel']['sha256'] for x in state['candidate_routing_records'])
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in (args.math_results/'source_snapshot').glob('*.py'):
        (snapshot/path.name).write_bytes(path.read_bytes())
    (snapshot/'benchmark_throughput_serve.py').write_bytes(benchmark.read_bytes())
    total=float(summary['Total tok/s']);ttft=float(summary['Mean TTFT (ms)'])
    values=[float(x['Total tok/s']) for x in rows[1:]]
    result={'complete':True,'official_benchmark':True,'audited_utc':datetime.now(timezone.utc).isoformat(),
        'total_tok_s':total,'output_tok_s':float(summary['Output tok/s']),'ttft_ms':ttft,
        'math_correct':quality['correct'],'math_total':105,'math_accuracy':quality['accuracy'],
        'requests_successful':512,'excluded_runs':[1],
        'previous_best_total_tok_s':args.previous_best_tok_s,'previous_best_ttft_ms':args.previous_best_ttft_ms,
        'vs_previous_best_throughput_percent':100*(total/args.previous_best_tok_s-1),
        'vs_previous_best_ttft_percent':100*(ttft/args.previous_best_ttft_ms-1),
        'vs_public_throughput_percent':100*(total/915.15-1),
        'vs_public_ttft_percent':100*(ttft/599262.03-1),
        'reference_label':'public16K baseline','workload':[16384,1024,64,128],
        'postfirst_spread_percent':100*(max(values)-min(values))/statistics.mean(values),
        'adopt_as_best':total>args.previous_best_tok_s,'profile':f"Hybrid +adaptive Decode{' KV16/w4/s1' if is_decode else ''} +native GQA Prefill M256/KV128/w16/s1, L0, budget{state['max_batched_tokens']}",
        'candidate_hook':quality['candidate_hook'],'experimental_kernel':quality['experimental_kernel'],
        'buckets':state.get('buckets','0'),'gather':state.get('gather','0'),
        'bucket_sources':quality.get('bucket_sources'),'bucket_trace_hook':quality.get('bucket_trace_hook'),
        'optimized_full_16k_not_run':False}
    if state.get('buckets')=='1':result['profile']+=' +75point Prefill buckets +fixed last-token gather'
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 天数原生 GQA 预填充正式16K结果','',
        f"完整四轮排除首轮，均值{total:.2f}total tok/s、{result['output_tok_s']:.2f}output tok/s、TTFT{ttft:.2f}ms。对应独立Math{quality['correct']}/105（{quality['accuracy']*100:.6f}%），严格>95%。四轮均128/128成功，总计512请求。",'',
        '|轮次|总tok/s|TTFT/ms|duration/s|纳入均值|','|---|---:|---:|---:|---|']
    lines.extend(f"|{i+1}|{x['Total tok/s']}|{x['Mean TTFT (ms)']}|{x['Benchmark Duration (s)']}|{'否' if i==0 else '是'}|" for i,x in enumerate(rows))
    lines+=['',f"相对先前正式最佳{args.previous_best_tok_s:.2f}tok/s，吞吐{result['vs_previous_best_throughput_percent']:+.6f}%；相对公开16K基线915.15tok/s，吞吐{result['vs_public_throughput_percent']:+.6f}%。后3轮极差/均值{result['postfirst_spread_percent']:.6f}%。",'',
        f"安装源码未改；与独立质量评测同进程钩子和预算。{'Decode保留原KV16分块、kernel body与grid，仅pipeline stage3改stage1；独立质量已通过。' if is_decode else 'Decode保持原配置。'} 本次为完整16K/1K、C64/N128原始四轮流程。"]
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
