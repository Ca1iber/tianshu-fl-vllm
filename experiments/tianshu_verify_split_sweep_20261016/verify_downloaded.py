# SPDX-License-Identifier: Apache-2.0
"""Recreate segment variants and recalculate all sweep event medians."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
from variant import render


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root.resolve()
    d=json.loads((root/'result.json').read_text());assert d['phase']=='complete' and d['complete']
    repo=Path(__file__).resolve().parents[2]
    for name,h in d['source_snapshots'].items():
        assert hashlib.sha256((root/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==h
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            assert hashlib.sha256((repo/Path(name).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==h
    raw=(root/'source_snapshot/workspace/vllm-plugin-FL/experiments/tianshu_ngram_framework_20261016/native_verify.py').read_bytes()
    for s in (4,8,16,32):
        text,derivation=render(raw,s);actual=d['variant_derivations'][str(s)]
        assert all(actual[k]==v for k,v in derivation.items())
        assert (root/'variant_generated'/f'generated_s{s}.py').read_text()==text
    modes=('original','s4','s8','s16','s32')
    assert len(d['cases'])==8 and {(r['case'],r['phase']) for r in d['cases']}=={(c,i) for c in range(4) for i in (0,1)}
    assert len(d['events'])==280 and {(r['case'],r['phase'],r['mode'],r['trial']) for r in d['events']}=={(c,i,m,t) for c in range(4) for i in (0,1) for m in modes for t in range(7)}
    assert all(r['ms']>0 for r in d['events'])
    for row in d['cases']:
        assert row['n']==64 and row['output_tail_preserved'] and set(row['checks'])==set(modes)
        assert all(r['passed'] for r in row['checks'].values()) and all(r['passed'] for r in row['CPU_checks'])
        assert row['changed_Graph_inputs']==bool(row['phase'])
        assert len(row['CPU_checks'])==sum(row['qlens'][i]*4 for i in (0,32,63))
        medians={m:statistics.median(r['ms'] for r in d['events'] if (r['case'],r['phase'],r['mode'])==(row['case'],row['phase'],m)) for m in modes}
        assert row['medians_ms']==medians and row['time_delta_percent']==100*(medians['s16']/medians['s8']-1)
    assert 'No running processes found' in (root/'hardware_after_process_exit.txt').read_text()
    report=dict(complete=True,all8_phases_40_FP32_checks=True,CPU_head_checks=sum(len(r['CPU_checks']) for r in d['cases']),all280_event_groups=True,
        source_snapshots_verified=len(d['source_snapshots']),all4_variants_recreated=True,
        full_raw_GPU_buffers_not_serialized=True,adopted=False,model_speedup_not_measured=True)
    (root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
