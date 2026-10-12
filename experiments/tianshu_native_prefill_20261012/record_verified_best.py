# SPDX-License-Identifier: Apache-2.0
"""Record a local best only after full official and Math archives are verified."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def main():
    parser=argparse.ArgumentParser()
    for name in ('official-root','official-manifest','official-archive',
                 'math-root','math-manifest','math-archive'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    manifests=[]
    for manifest_path, archive in [(args.official_manifest,args.official_archive),
                                   (args.math_manifest,args.math_archive)]:
        manifest=json.loads(manifest_path.read_text())
        assert archive.stat().st_size==manifest['bytes']
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['sha256']
        manifests.append(manifest)
    official=args.official_root.resolve()
    quality=args.math_root.resolve()
    result=json.loads((official/'result.json').read_text())
    audit=json.loads((official/'comparison.json').read_text())
    math=json.loads((quality/'accuracy_audit.json').read_text())
    assert result['phase']=='complete' and result['gpu_idle_after']
    assert result['benchmark_exit_code']==0
    assert result['completed_runs']==result['runs']==4 and result['skip_first']==1
    assert audit['complete'] and audit['requests_successful']==1024
    assert math['complete'] and math['total']==105 and math['accuracy']>.95
    assert not math['reused_original_accuracy'] and math['experimental_candidate']=='native-prefill'
    assert math['execution_summary']['succeeded']==105 and math['execution_summary']['errored']==0
    assert result['max_batched_tokens']==math['max_batched_tokens']==2048
    assert audit['math_correct']==math['correct'] and audit['math_accuracy']==math['accuracy']
    for key in ('worker_sha256','adaptive_sha256','hybrid_sha256'):
        assert result[key]==math[key]
    for relative, info in math['raw_files'].items():
        p=quality/relative
        assert p.stat().st_size==info['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
    assert result['candidate_hook']==math['candidate_hook']
    assert result['experimental_kernel']==math['experimental_kernel']
    generated={r['generated_kernel_sha256'] for r in math['candidate_routing_records']}
    assert all(r['generated_kernel_sha256'] in generated for r in result['candidate_routing_records'])
    assert hashlib.sha256((official/'source_snapshot/candidate_sitecustomize.py').read_bytes()).hexdigest()==math['candidate_hook']['sha256']
    assert hashlib.sha256((official/'source_snapshot/zero_initialization.py').read_bytes()).hexdigest()==math['experimental_kernel']['sha256']
    csv_paths=list((official/'benchmark_results').glob('raw_runs_*.csv'))
    assert len(csv_paths)==1
    rows=list(csv.DictReader(csv_paths[0].open()))
    assert rows==result['raw_runs'] and len(rows)==4
    for r in rows:
        assert r['Run Status']=='SUCCESS' and int(float(r['Successful Requests']))==256
        assert [int(r[k]) for k in ('Prefill','Decode','Conc','Num Prompts')]==[4096,1024,64,256]
        assert int(float(r['Total Input Tokens']))==1048576
        assert int(float(r['Total Output Tokens']))==262144
    for field, key in [('Total tok/s','total_tok_s'),('Output tok/s','output_tok_s'),
                       ('Mean TTFT (ms)','ttft_ms')]:
        assert abs(statistics.mean(float(r[field]) for r in rows[1:])-audit[key])<.011
    assert hashlib.sha256((official/'source_snapshot/benchmark_throughput_serve.py').read_bytes()).hexdigest()=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
    workspace=Path(__file__).resolve().parents[3]
    service=workspace/'service-results'
    best_path=service/'20261012_tianshu_current_best.json'
    previous=json.loads(best_path.read_text())
    assert audit['total_tok_s']>previous['official_4k_total_tok_s'], 'Not an improvement over retained best'
    now=datetime.now(timezone(timedelta(hours=8))).isoformat()
    best=dict(as_of_local=now,tag='tianshu-best-native-prefill-20261012',commit='pending_local_checkpoint',
              profile=audit['profile'],official_4k_total_tok_s=audit['total_tok_s'],
              official_4k_output_tok_s=audit['output_tok_s'],official_4k_mean_ttft_ms=audit['ttft_ms'],
              official_requests_successful=1024,official_runs=4,excluded_runs=[1],
              math_level3_correct=math['correct'],math_level3_total=105,math_accuracy=math['accuracy'],
              official_accuracy_requirement='>95%',benchmark_script_unmodified=True,
              official_archive_sha256=manifests[0]['sha256'],math_archive_sha256=manifests[1]['sha256'],
              both_archives_verified_locally=True,postfirst_spread_percent=audit['postfirst_spread_percent'],
              vs_public_throughput_percent=audit['vs_public_throughput_percent'],
              vs_public_ttft_percent=audit['vs_public_ttft_percent'],
              launcher='/workspace/vllm-plugin-FL/experiments/tianshu_native_prefill_20261012/serve_tianshu_minicpm.sh',
              full_16k_optimized_profile_not_benchmarked=True,ongoing_optimization=True,
              gpu_work_stop_local=previous['gpu_work_stop_local'],lease_end_local=previous['lease_end_local'],
              previous_best=dict(tag=previous['tag'],commit=previous['commit'],
                                 total_tok_s=previous['official_4k_total_tok_s'],
                                 ttft_ms=previous['official_4k_mean_ttft_ms']))
    best_path.write_text(json.dumps(best,indent=2)+'\n')
    target=Path(__file__).resolve().parent/'results/20261012/official'
    target.mkdir(parents=True,exist_ok=True)
    for name in ('comparison.json','README.md'):
        shutil.copy2(official/name,target/name)
    for p in (official/'benchmark_results').glob('*.csv'):
        shutil.copy2(p,target/p.name)
    (target/'backup_manifest.json').write_text(json.dumps(manifests[0],indent=2)+'\n')
    checkpoint=dict(best)
    checkpoint.pop('commit')
    checkpoint['commit_resolution']='git rev-parse tianshu-best-native-prefill-20261012^{}'
    (target/'best_checkpoint.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
    print(json.dumps(best,indent=2))


if __name__=='__main__':
    main()
