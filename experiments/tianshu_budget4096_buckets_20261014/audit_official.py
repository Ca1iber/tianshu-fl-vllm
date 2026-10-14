# SPDX-License-Identifier: Apache-2.0
"""Audit only complete official runs and their matching independent quality."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from audit_profile import audit_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--math-results',type=Path,required=True)
    parser.add_argument('--previous-best-tok-s',type=float,default=3826.24)
    parser.add_argument('--previous-best-ttft-ms',type=float,default=4457.42)
    args = parser.parse_args()
    root = args.output.resolve()
    state = json.loads((root/'result.json').read_text())
    quality = json.loads((args.math_results/'accuracy_audit.json').read_text())
    assert state['phase']=='complete' and state['gpu_idle_after'] and state['benchmark_exit_code']==0
    assert state['completed_runs']==state['runs']==4 and state['skip_first']==1
    assert quality['complete'] and quality['total']==105 and quality['accuracy']>.95
    assert not quality['reused_original_accuracy'] and quality['experimental_candidate']==state['experimental_candidate']
    assert state['fresh_math_for_B64S8_segments'] and quality['fresh_math_for_B64S8_segments']
    assert state['candidate_hook']==quality['candidate_hook']
    assert state['experimental_kernel']==quality['experimental_kernel']
    assert state.get('buckets','0')==quality.get('buckets','0')
    assert state.get('gather','0')==quality.get('gather','0')
    assert state['max_batched_tokens']==quality['max_batched_tokens']
    if state.get('buckets')=='1':
        assert state['bucket_sources']==quality['bucket_sources']
        assert state['bucket_trace_hook']==quality['bucket_trace_hook']
        assert state['actual_gather_records'] and any(x['index_dtype']=='torch.int32' for x in state['actual_gather_records'])
        assert any(x['complete'] and len(x['calls'])==83 for x in state['bucket_warmup_records'])
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
    is_decode=state['experimental_candidate']=='decode-coherent'
    quality_native=quality['native_prefill_routing_records'] if is_decode else quality['candidate_routing_records']
    official_native=state['native_prefill_routing_records'] if is_decode else state['candidate_routing_records']
    generated={x['generated_kernel_sha256'] for x in quality_native}
    assert official_native and all(x['generated_kernel_sha256'] in generated for x in official_native)
    if is_decode:
        assert state['candidate_routing_records']
        for x in state['candidate_routing_records']:
            expected_segments=8 if x['batch']==64 else 64 if x['batch']==1 else 4 if 16<=x['batch']<=64 else 16
            assert x['segments']==expected_segments
            tile = 16 if x['batch'] in (8,16,32,40,48,56,64) else 32
            warps = 1 if tile == 16 else 2 if 8 <= x['batch'] <= 64 else 4
            assert x['configuration'] == dict(TILE_SIZE=tile, num_warps=warps, num_stages=1)
            assert x['compiled_metadata']['num_warps'] == warps
            assert x['generated_source_sha256'] in {a['generated_source_sha256'] for a in quality['candidate_routing_records']}
        assert state['fresh_math_for_decode_kv16_w1'] and quality['fresh_math_for_decode_kv16_w1']
        assert state['segment_sources'] == quality['segment_sources']
        assert state['server_command'] == quality['server_command']
        assert all(x['configuration'] == dict(num_warps=16, num_stages=1) for x in state['topp_launch_records'])
        assert {(x['batch'], x['segments']) for x in state['candidate_routing_records']} == {(x['batch'], x['segments']) for x in state['decode_coherent_reducer_records']}
        assert state['fresh_math_for_coherent_decode'] and quality['fresh_math_for_coherent_decode']
        assert state['decode_coherent_reducer_records'] and all(x['actual_tile']==(16 if x['batch'] in (8,16,32,40,48,56,64) else 32) for x in state['decode_coherent_reducer_records'])
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    assert len(state['formal_sources'])==5 and len(quality['segment_sources'])==10
    for path,sha in state['formal_sources'].items():
        source=Path(path)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
        assert hashlib.sha256((root/'official_source_snapshot'/source.relative_to('/workspace/vllm-plugin-FL/experiments')).read_bytes()).hexdigest()==sha
    policy=state['segment_policy_records']
    assert policy and any(x['batch']==64 and x['baseline_segments']==4 and x['selected_segments']==8 for x in policy)
    assert all(x['selected_segments']==(8 if x['batch']==64 else x['baseline_segments']) for x in policy)
    for path in (args.math_results/'source_snapshot').rglob('*.py'):
        destination=snapshot/path.relative_to(args.math_results/'source_snapshot')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(path.read_bytes())
    profile = audit_profile(root, state)
    (snapshot/'benchmark_throughput_serve.py').write_bytes(benchmark.read_bytes())
    total=float(summary['Total tok/s']);ttft=float(summary['Mean TTFT (ms)'])
    values=[float(x['Total tok/s']) for x in rows[1:]]
    result={'complete':True,'official_benchmark':True,'audited_utc':datetime.now(timezone.utc).isoformat(),
        'total_tok_s':total,'output_tok_s':float(summary['Output tok/s']),'ttft_ms':ttft,
        'math_correct':quality['correct'],'math_total':105,'math_accuracy':quality['accuracy'],
        'requests_successful':1024,'excluded_runs':[1],
        'previous_best_total_tok_s':args.previous_best_tok_s,'previous_best_ttft_ms':args.previous_best_ttft_ms,
        'vs_previous_best_throughput_percent':100*(total/args.previous_best_tok_s-1),
        'vs_previous_best_ttft_percent':100*(ttft/args.previous_best_ttft_ms-1),
        'vs_public_throughput_percent':100*(total/2028.01-1),
        'vs_public_ttft_percent':100*(ttft/11573.53-1),
        'vs_original_measured_throughput_percent':100*(total/1876.67-1),
        'postfirst_spread_percent':100*(max(values)-min(values))/statistics.mean(values),
        'adopt_as_best':total>args.previous_best_tok_s,'profile':f"Hybrid +adaptive Decode{' KV16/w1/s1 L0 for batches8/16/32/40/48/56/64; otherwise KV32/w2 for batch8..64 and w4 for smaller batches' if is_decode else ''} +native GQA Prefill M256/KV128/w16/s1, L0, budget{state['max_batched_tokens']}",
        'fresh_math_for_coherent_decode':True,
        'decode_coherent_reducer_records':state['decode_coherent_reducer_records'],
        'candidate_hook':quality['candidate_hook'],'experimental_kernel':quality['experimental_kernel'],
        'buckets':state.get('buckets','0'),'gather':state.get('gather','0'),
        'bucket_sources':quality.get('bucket_sources'),'bucket_trace_hook':quality.get('bucket_trace_hook'),
        'optimized_full_16k_not_run':True}
    result['fresh_math_for_B64S8_segments'] = True
    result['budget4096_profile'] = profile
    result['formal_sources'] = state['formal_sources']
    result['segment_policy_records'] = state['segment_policy_records']
    result['fresh_math_for_decode_kv16_w1'] = True
    result['segment_sources'] = quality['segment_sources']
    result['topp_launch_records'] = state['topp_launch_records']
    result['profile'] += ' +B64 segments8 (others unchanged) +nativeTopP w16/s1'
    if state.get('buckets')=='1':result['profile']+=' +83point Prefill buckets +fixed last-token gather +target-table SQL reflection'
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 天数 guarded KV16/w1、B64 segments8 正式4K结果','',
        f"完整四轮排除首轮，均值{total:.2f}total tok/s、{result['output_tok_s']:.2f}output tok/s、TTFT{ttft:.2f}ms。对应独立Math{quality['correct']}/105（{quality['accuracy']*100:.6f}%），严格>95%。四轮均256/256成功，总计1024请求。",'',
        '|轮次|总tok/s|TTFT/ms|duration/s|纳入均值|','|---|---:|---:|---:|---|']
    lines.extend(f"|{i+1}|{x['Total tok/s']}|{x['Mean TTFT (ms)']}|{x['Benchmark Duration (s)']}|{'否' if i==0 else '是'}|" for i,x in enumerate(rows))
    lines+=['',f"相对先前正式最佳{args.previous_best_tok_s:.2f}tok/s，吞吐{result['vs_previous_best_throughput_percent']:+.6f}%；相对公开2028.01tok/s，吞吐{result['vs_public_throughput_percent']:+.6f}%。后3轮极差/均值{result['postfirst_spread_percent']:.6f}%。",'',
        f"安装源码未改；与独立质量评测同进程钩子和预算。{'Decode使用KV16/w1/stage1（batch8/16/32/40/48/56/64；B24等保留KV32/w2，小batch保留KV32/w4）、无sink L0，producer和reducer TILE一致，实际路由与独立质量匹配。' if is_decode else 'Decode配置见实际路由记录。'} 未运行优化版完整16K，不将短测或算子耗时当作官方分数。"]
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
