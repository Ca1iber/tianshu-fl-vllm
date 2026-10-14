# SPDX-License-Identifier: Apache-2.0
"""Recompute all passing tile measurements and retain every failed candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    data=json.loads((root/'result.json').read_text())
    assert data['complete'] and not data['official_benchmark'] and not data['adopted']
    assert not data['tf32'] and not data['bf16_reduced_precision_reduction']
    assert not data['installed_sources_modified'] and data['no_flaggems_aten_registrar']
    assert len(data['frozen_sources'])==3
    for i,(source,digest) in enumerate(data['frozen_sources'].items()):
        snapshot=root/'source_snapshot'/f'{i:02d}_{Path(source).name}'
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==digest
        if i<2:
            local=Path(__file__).resolve().parents[2]/Path(source).relative_to('/workspace/vllm-plugin-FL')
            assert hashlib.sha256(local.read_bytes()).hexdigest()==digest
    assert len(data['cases'])==40 and len(data['configs'])==10
    passing=[];failures=[];shapes=[('qkv',2560,2048),('o',2048,2048),('gateup',12288,2048),('down',2048,6144)]
    for shape_i,(name,n,k) in enumerate(shapes):
        for config_i,config in enumerate(data['configs']):
            item=data['cases'][shape_i*10+config_i]
            assert (item['projection'],item['M'],item['N'],item['K'])==(name,4096,n,k)
            assert item['config']==config
            assert config['splits']==1 and config['group_m'] in (1,8)
            if not item.get('numerical_passed'):
                assert item.get('error') and 'raw_ms' not in item
                failures.append(item);continue
            assert 'error' not in item and item['changed_Graph_inputs_checked']
            for group in ('fp32_checks','cpu_checks'):
                assert len(item[group])==2
                for check in item[group]:
                    assert check['passed'] and check['failed_elements']==0
                    assert check['atol']==.02 and check['rtol']==.01
            raw=item['raw_ms']
            assert len(raw)==2 and all(len(v)==7 and all(x>0 for x in v) for v in raw)
            medians=list(map(statistics.median,raw));assert medians==item['median_ms']
            change=100*(medians[1]/medians[0]-1)
            assert abs(change-item['time_change_percent'])<1e-10
            passing.append(dict(projection=name,config=config,baseline_ms=medians[0],candidate_ms=medians[1],time_change_percent=change))
    assert len(passing)==data['numerically_passing_cases']
    assert 'No running processes found' in (root/'hardware_after.txt').read_text()
    best={name:min((c for c in passing if c['projection']==name),key=lambda c:c['time_change_percent'],default=None) for name,_,_ in shapes}
    audit=dict(complete=True,official_benchmark=False,adopted=False,source3_verified=True,
               total_cases=40,passing_cases=len(passing),failed_cases=len(failures),
               changed_Graph_GPU_FP32_and_CPU_sample_checks_verified=True,
               raw_event_values_recomputed=14*len(passing),best_by_projection=best,rows=passing,
               failures=failures,scope='Synthetic M4096 tiles only; no model throughput or quality claim')
    (root/'local_download_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('rows','failures')}))


if __name__=='__main__':main()
