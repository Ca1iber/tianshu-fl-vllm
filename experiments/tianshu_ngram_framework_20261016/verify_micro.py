# SPDX-License-Identifier: Apache-2.0
"""Verify downloaded compatibility records and recalculate attention medians."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root.resolve()
    def read(path):return json.loads(path.read_text())
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    repo=Path(__file__).resolve().parents[2]
    names={'rejection':'20261016_ngram_rejection_micro_tianshu','failed_attention':'20261016_ngram_attention_micro_tianshu','attention':'20261016_ngram_attention_v2_micro_tianshu'}
    results={k:read(root/v/'result.json') for k,v in names.items()}
    for key,d in results.items():
        for source,h in d['source_snapshots'].items():
            assert sha(root/names[key]/'source_snapshot'/Path(source).relative_to('/'))==h
            if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
                assert sha(repo/Path(source).relative_to('/workspace/vllm-plugin-FL'))==h
    reject=results['rejection'];assert reject['complete'] and reject['fixed_all_passed'] and len(reject['fixed_cases'])==48
    for row in reject['cases']:
        assert row['bitwise_cpu_equal']==(row['gpu_observed']==row['cpu_expected'])
        assert row['bitwise_cpu_equal']==(row['uniform_dtype']=='torch.float32')
    assert reject['fp64_uniform_host_min']==reject['fp64_uniform_host_max']==0 and reject['fp64_uniform_unique']==1
    for row in reject['fixed_cases']:assert row['bitwise_cpu_equal'] and 0<row['min_threshold']<=row['max_threshold']<=1
    assert {(tuple(r['counts']),r['seed'],r['no_draft_probs'],r['mixed_greedy']) for r in reject['fixed_cases']}=={(tuple(c),seed,nd,mg) for c in ([7],[3,7],[0,1,7],[7]*64) for seed in (17,73,139) for nd in (False,True) for mg in (False,True)}
    summary=reject['fixed_uniform_summary'];assert .49<summary['mean']<.51 and summary['unique']>60000
    failed=results['failed_attention'];assert failed['phase']=='failed' and not failed['complete'] and len(failed['cases'])==0
    d=results['attention'];assert d['phase']=='complete' and d['complete'] and len(d['cases'])==14
    assert len(d['events'])==196
    assert {(r['case'],r['phase']) for r in d['cases']}=={(c,i) for c in range(7) for i in (0,1)}
    assert {(r['case'],r['phase'],r['mode'],r['trial']) for r in d['events']}=={(c,i,m,t) for c in range(7) for i in (0,1) for m in ('original','split') for t in range(7)}
    assert all(r['ms']>0 for r in d['events'])
    for row in d['cases']:
        assert len(row['qlens'])==len(row['lengths'])==row['n'] and row['output_tail_preserved']
        assert row['changed_Graph_inputs']==bool(row['phase'])
        assert set(row['checks'])=={'original','split'} and all(c['passed'] for c in row['checks'].values())
        assert all(c['passed'] for c in row['CPU_checks'])
        expected=sum(row['qlens'][i]*4 for i in {0,row['n']//2,row['n']-1})
        assert len(row['CPU_checks'])==expected
        medians={m:statistics.median(r['ms'] for r in d['events'] if (r['case'],r['phase'],r['mode'])==(row['case'],row['phase'],m)) for m in ('original','split')}
        assert medians==row['medians_ms'] and row['time_delta_percent']==100*(medians['split']/medians['original']-1)
    spec=importlib.util.spec_from_file_location('_verify_ngram_render',Path(__file__).with_name('native_verify.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    raw=root/names['attention']/'source_snapshot/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py'
    text,derivation=module.render(raw.read_bytes())
    assert (root/names['attention']/'split_generated/generated_split_unified.py').read_text()==text
    split=[r for r in d['routes'] if r['route']=='Q2_split'];assert split
    assert all(r['derivation']==derivation and r['scratch_capacity_tokens']==128 and r['segments'] in (8,16) for r in split)
    result=dict(complete=True,all48_rejection_cases=True,all14_attention_phases=True,all196_event_groups=True,
        all_frozen_sources_and_generated_dispatch_checked=True,full_raw_GPU_buffers_not_serialized=True,
        raw_native_FP64_and_L1_failures_preserved=True,
        remote_results={k:dict(path='/workspace/benchmark-results/'+names[k]+'/result.json',sha256=sha(root/names[k]/'result.json')) for k in ('rejection','attention')},
        attention_component_only=True,model_throughput_and_new_Math_not_measured=True,adopted=False)
    (root/'local_download_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
