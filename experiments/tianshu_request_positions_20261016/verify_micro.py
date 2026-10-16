# SPDX-License-Identifier: Apache-2.0
"""Independently audit downloaded method artifacts, records and event medians."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root.resolve();d=json.loads((root/'result.json').read_text())
    assert d['phase']=='complete' and d['complete'] and not d['official_benchmark'] and not d['adopted']
    assert len(d['source_snapshots'])==6
    repo=Path(__file__).resolve().parents[2]
    for source,sha in d['source_snapshots'].items():
        assert hashlib.sha256((root/'source_snapshot'/Path(source).relative_to('/')).read_bytes()).hexdigest()==sha
        if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
            assert hashlib.sha256((repo/Path(source).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha
    spec=importlib.util.spec_from_file_location('_verify_request_positions_render',Path(__file__).with_name('render.py'))
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    raw=(root/'source_snapshot/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py').read_bytes()
    for mode in ('control','int32','fused'):
        methods,derivation=renderer.render(raw,mode);assert derivation==d['derivations'][mode]
        for name,text in methods.items():assert text==(root/(mode+'_'+name+'.py')).read_text()
    shapes=((1,1),(8,8),(29,29),(30,30),(32,32),(64,64),(64,290),(64,565),(64,1323),(64,1345),(64,1537),(64,2048),(128,2048),(256,2048))
    assert len(d['cases'])==56
    assert {(c['n'],c['m'],c['phase'],c['mode']) for c in d['cases']}=={(n,m,i,mode) for n,m in shapes for i in (0,1) for mode in ('int32','fused')}
    keys={'positions','seq_lens','num_computed_tokens','logits','no_spec','real_slot_arguments','int64_positions'}
    for name in ('input_ids','query_start_loc','req_indices','query_pos','num_scheduled_tokens','discard_request_mask','num_accepted_tokens','prev_positions'):
        keys.update((name+'_cpu',name+'_gpu'))
    for case in d['cases']:
        assert set(case['checks'])==keys and all(v is True for v in case['checks'].values()) and case['independent_CPU_positions']
    assert len(d['graphs'])==8
    assert {(g['n'],g['m'],g['turn']) for g in d['graphs']}=={(n,m,t) for n,m in ((64,64),(64,290),(64,1537),(256,2048)) for t in (0,1)}
    assert all(g['changed_computed_and_query'] and g['cleared_outputs'] and g['CPU_equal'] for g in d['graphs'])
    assert len(d['timings'])==63 and len(d['cold'])==42
    assert {(r['n'],r['m'],r['mode'],r['repeat']) for r in d['timings']}=={(64,m,mode,i) for m in (64,290,1537) for mode in ('control','int32','fused') for i in range(7)}
    assert all(r['host_ms']>0 and r['stream_ms']>0 and r['iterations']==20 for r in d['timings'])
    medians={str(m):{mode:{k:statistics.median(r[k] for r in d['timings'] if r['m']==m and r['mode']==mode) for k in ('host_ms','stream_ms')} for mode in ('control','int32','fused')} for m in (64,290,1537)}
    assert medians==d['medians']
    # Captured inside the probe's finally block: the driver may omit the
    # literal empty-table message until the process's CUDA context exits.
    hardware=(root/'hardware_after.txt').read_text()
    assert not re.findall(r'^\|\s+\d+\s+\d+\s+\S+',hardware,re.M)
    result=dict(complete=True,all6_sources_checked=True,all56_method_checks=True,all8_changed_graph_checks=True,
        all63_event_records=True,generated_methods_reproduced=True,slot_mapping_observer_stub=True,
        full_buffer_values_available_for_independent_recompute=False,medians=medians,
        remote_result='/workspace/benchmark-results/'+root.name+'/result.json',
        remote_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
        hardware_after_has_no_listed_GPU_processes=True,
        next_service_requires_fresh_post_exit_idle_check=True,
        official_benchmark=False,model_speedup_not_measured=True)
    (root/'local_download_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
