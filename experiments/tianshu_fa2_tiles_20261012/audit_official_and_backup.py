# SPDX-License-Identifier: Apache-2.0
"""Audit a completed tile official trial and archive its complete evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--math-results',type=Path,required=True)
    parser.add_argument('--previous-official',type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve()
    state=json.loads((root/'result.json').read_text())
    previous=json.loads((args.previous_official/'result.json').read_text())
    math=json.loads((args.math_results/'accuracy_audit.json').read_text())
    assert state['phase']==previous['phase']=='complete'
    assert state['gpu_idle_after'] and state['benchmark_exit_code']==0
    assert state['completed_runs']==state['runs']==4 and state['skip_first']==1
    assert math['complete'] and math['total']==105 and math['accuracy']>.95
    assert state['independent_math']['correct']==math['correct']
    assert state['tile_hook']['sha256']==math['tile_hook']['sha256']
    assert all(record['configuration']==state['tile_configuration'] for record in state['tile_routing_records'])
    rows=state['raw_runs']
    assert len(rows)==4
    for row in rows:
        assert row['Run Status']=='SUCCESS' and float(row['Successful Requests'])==256
        assert [int(row[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
        assert float(row['Total Input Tokens'])==1048576 and float(row['Total Output Tokens'])==262144
    summary=state['official_summary'];before=previous['official_summary']
    for key in ('Total tok/s','Output tok/s','Mean TTFT (ms)'):
        assert abs(float(summary[key])-statistics.mean(float(r[key]) for r in rows[1:]))<.011
    bench=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
    assert sha(bench)==state['official_script_sha256']=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
    snapshots=root/'source_snapshot';snapshots.mkdir(exist_ok=True)
    for filename,key in (('worker.py','worker_sha256'),('iluvatar_hybrid_attention.py','hybrid_sha256'),
                         ('iluvatar_adaptive_decode.py','adaptive_sha256')):
        source=args.math_results/'source_snapshot'/filename
        assert sha(source)==state[key]
        (snapshots/filename).write_bytes(source.read_bytes())
    hook=args.math_results/'source_snapshot/tile_sitecustomize.py'
    assert sha(hook)==state['tile_hook']['sha256']
    (snapshots/'tile_sitecustomize.py').write_bytes(hook.read_bytes())
    (snapshots/'benchmark_throughput_serve.py').write_bytes(bench.read_bytes())
    values=[float(r['Total tok/s']) for r in rows[1:]]
    total=float(summary['Total tok/s']);ttft=float(summary['Mean TTFT (ms)'])
    result={'audited_at_utc':datetime.now(timezone.utc).isoformat(),'complete':True,
            'official_benchmark':True,'successful_requests':1024,'excluded_runs':[1],
            'total_tok_s':total,'output_tok_s':float(summary['Output tok/s']),'ttft_ms':ttft,
            'math_correct':math['correct'],'math_total':105,'math_accuracy':math['accuracy'],
            'quality_requirement_passed':True,'postfirst_spread_percent':100*(max(values)-min(values))/statistics.mean(values),
            'previous_official_dir':str(args.previous_official),
            'vs_previous_throughput_percent':100*(total/float(before['Total tok/s'])-1),
            'vs_previous_ttft_percent':100*(ttft/float(before['Mean TTFT (ms)'])-1),
            'vs_original_measured_throughput_percent':100*(total/1876.67-1),
            'vs_original_measured_ttft_percent':100*(ttft/12315.86-1),
            'vs_public_throughput_percent':100*(total/2028.01-1),
            'vs_public_ttft_percent':100*(ttft/11573.53-1),
            'combined_profile':'Hybrid1 + adaptive1 + FA2 tile128 + budget2048',
            'isolated_tile_official_effect_established':False,
            'short_pairs_isolate_tile_effect':True,
            'adopt_as_best':total>float(before['Total tok/s']) and ttft<float(before['Mean TTFT (ms)']),
            'raw_files':{str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)}
                         for p in (root/'result.json',*sorted((root/'benchmark_results').glob('*.csv')))}}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 天数FA2 tile128组合配置正式4K结果','',
           f"结论：正式四轮排除首轮，均值{total:.2f} total tok/s、{result['output_tok_s']:.2f} output tok/s、TTFT{ttft:.2f}ms。对应独立Math-500 Level3为{math['correct']}/105（{100*math['accuracy']:.6f}%），满足严格>95%。四轮全部256/256成功，共1024请求。",'',
           '|轮次|总tok/s|TTFT/ms|duration/s|纳入均值|','|---|---:|---:|---:|---|']
    lines.extend(f"|{i+1}|{r['Total tok/s']}|{r['Mean TTFT (ms)']}|{r['Benchmark Duration (s)']}|{'否' if i==0 else '是'}|" for i,r in enumerate(rows))
    lines.extend(['',f"相对先前正式Hybrid配置，吞吐{result['vs_previous_throughput_percent']:+.4f}%、TTFT{result['vs_previous_ttft_percent']:+.4f}%；最后三轮吞吐极差/均值{result['postfirst_spread_percent']:.4f}%。采用为当前最佳：{result['adopt_as_best']}。",'',
                  '此正式对比包含adaptive和tile两项相对先前Hybrid配置的变化，不声称正式成绩隔离了tile效果。较大短对照只改变tile得到+9.4847%，这些短测不作为官方分数。', '',
                  '官方benchmark脚本SHA与原始脚本一致；全部计时、源快照、配置路由、原始CSV和日志随备份保留。仅验证官方4K，不运行完整16K。'])
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    archive=Path('/tmp/tianshu_fa2_tile_official_complete.tgz')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(root,arcname=root.name)
    data=archive.read_bytes();parts=[]
    for index,offset in enumerate(range(0,len(data),262144)):
        part=archive.with_name(archive.name+f'.part{index:03d}');chunk=data[offset:offset+262144];part.write_bytes(chunk)
        parts.append({'name':part.name,'bytes':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()})
    manifest={'archive':str(archive),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'parts':parts}
    Path('/tmp/tianshu_fa2_tile_official_complete_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'comparison':result,'backup':manifest},indent=2))


if __name__=='__main__':
    main()
