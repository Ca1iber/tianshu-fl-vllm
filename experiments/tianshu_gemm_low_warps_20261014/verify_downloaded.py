# SPDX-License-Identifier: Apache-2.0
"""Audit all actually executed low-warp GEMM configurations and timing samples."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();base=Path(__file__).resolve().parent
    result=json.loads((root/'result.json').read_text());assert result['complete']
    assert not result['official_benchmark'] and not result['adoption_allowed']
    assert not result['installed_files_modified']
    expected_source='215ea26ed636c59790b932445f6d2bb04d24f3427a699294791293788df3e570'
    assert sha(base/'probe_mm.py')==sha(root/'source_snapshot/probe_mm.py')==expected_source
    kernel=base.parent/'tianshu_splitk_20261012/experimental_mm.py'
    assert sha(kernel)==sha(root/'source_snapshot/experimental_mm.py')==result['source_sha256']=='86a2244f3fb30a8294d53174bd3aa6080f947b5b80ce43b6212a35b533463d56'
    assert sha(root/'source_snapshot/baseline_mm.py')==result['baseline_source']['sha256']
    rows=result['cases'];assert len(rows)==80
    shapes={'qkv':(2560,2048),'o_proj':(2048,2048),'gate_up':(12288,2048),'down':(2048,6144),'lm_head':(130560,2048)}
    configs={(16,64,32,1),(16,64,64,1),(32,64,32,2),(32,64,64,2),(64,64,32,2),(64,64,64,2),(32,128,32,2),(64,128,32,2)}
    seen=set();errors=[];numerical_failures=[];timed=[];summary=[]
    for row in rows:
        assert row['m'] in (16,64) and (row['n'],row['k'])==shapes[row['name']]
        c=row['config'];key=(c['bm'],c['bn'],c['bk'],c['warps'])
        assert key in configs and c['splits']==c['stages']==1
        identity=(row['m'],row['name'],key);assert identity not in seen;seen.add(identity)
        for name in ('baseline_vs_fp32','baseline_vs_cpu'):
            assert row[name]['passed'] and row[name]['finite']
        if 'error' in row:
            errors.append({'m':row['m'],'name':row['name'],'config':c,'error':row['error']});continue
        checks=[row[k] for k in ('vs_fp32','vs_cpu','vs_native')]
        assert all(x['atol']==.02 and x['rtol']==.01 for x in checks)
        if not all(x['passed'] and x['finite'] for x in checks):
            numerical_failures.append({'m':row['m'],'name':row['name'],'config':c,'checks':checks})
            assert 'raw_ms' not in row;continue
        assert row['graph_numeric_passed']
        assert len(row['raw_ms'])==2 and all(len(x)==7 for x in row['raw_ms'])
        before,after=map(statistics.median,row['raw_ms'])
        assert before>0 and after>0
        assert abs(before-row['baseline_ms'])<1e-9 and abs(after-row['candidate_ms'])<1e-9
        assert abs(100*(after/before-1)-row['change_percent'])<1e-8
        timed.append(row)
    assert len(seen)==80
    for m in (64,16):
        for name in shapes:
            choices=[x for x in timed if x['m']==m and x['name']==name]
            if choices:
                best=min(choices,key=lambda x:x['change_percent'])
                summary.append({k:best[k] for k in ('m','name','config','baseline_ms','candidate_ms','change_percent')})
    audit=dict(complete=True,original_shape_config_count=80,timed_configurations=len(timed),error_count=len(errors),numerical_failure_count=len(numerical_failures),
               all_numerical_configurations_passed=not errors and not numerical_failures,errors=errors,numerical_failures=numerical_failures,
               source_snapshots_verified=True,explicit_baseline_provider='FlagGems.mm',timed_configurations_passed_gpu_fp32_cpu_subset_native=True,
               timing_scope='7 alternating CUDA Graph timings,20 calls per replay; CPU reference subset8rows32columns, no whole-matrix CPU claim',
               official_benchmark=False,adopted=False,best_per_shape=summary)
    (root/'local_download_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
