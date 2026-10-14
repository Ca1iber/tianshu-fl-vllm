# SPDX-License-Identifier: Apache-2.0
"""Independently recompute the API diagnostic from downloaded raw events."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = json.loads((root/'result.json').read_text())
    assert data['complete'] and data['all_numerical_checks_passed']
    assert not data['official_benchmark'] and not data['adopted']
    assert not data['tf32'] and not data['bf16_reduced_precision_reduction']
    assert data['no_flaggems_aten_registrar'] and not data['installed_sources_modified']
    assert data['comparison'] == ['flag_gems.mm', 'torch.mm']
    assert len(data['frozen_sources']) == 3
    snapshots = sorted((root/'source_snapshot').iterdir())
    for i, (source, digest) in enumerate(data['frozen_sources'].items()):
        assert snapshots[i].name == f'{i:02d}_{Path(source).name}'
        assert hashlib.sha256(snapshots[i].read_bytes()).hexdigest() == digest
    repo = Path(__file__).resolve().parents[2]
    assert hashlib.sha256((repo/'experiments/tianshu_mm_backend_20261014/probe_mm.py').read_bytes()).hexdigest() == next(iter(data['frozen_sources'].values()))
    assert (root/'aten_mm_dispatch.txt').stat().st_size > 0
    assert 'No running processes found' in (root/'hardware_after.txt').read_text()
    projections = [('qkv',2560,2048),('o',2048,2048),('gateup',12288,2048),('down',2048,6144),('head',130560,2048)]
    expected = [(name,m,n,k) for m in (8,16,32,64) for name,n,k in projections]
    expected += [(name,2048,n,k) for name,n,k in projections if name != 'head']
    assert len(data['cases']) == len(expected) == 24
    rows = []
    for item, shape in zip(data['cases'], expected):
        assert tuple(item[k] for k in ('projection','M','N','K')) == shape
        assert 'error' not in item and item['numerical_passed'] and item['changed_Graph_inputs_checked']
        for group in ('fp32_checks','cpu_checks'):
            assert len(item[group]) == 2
            for check in item[group]:
                assert check['passed'] and check['failed_elements'] == 0
                assert check['atol'] == .02 and check['rtol'] == .01
        raw = item['raw_ms']
        assert len(raw) == 2 and all(len(v) == 7 and all(x > 0 for x in v) for v in raw)
        medians = list(map(statistics.median, raw))
        assert medians == item['median_ms']
        change = 100*(medians[1]/medians[0]-1)
        assert abs(change-item['time_change_percent']) < 1e-10
        rows.append(dict(projection=shape[0],M=shape[1],N=shape[2],K=shape[3],
                         flaggems_ms=medians[0],torch_ms=medians[1],time_change_percent=change))
    audit = dict(complete=True,official_benchmark=False,adopted=False,
                 scope='24 synthetic API diagnostic cases; no model binding or backend eligibility claim',
                 source3_verified=True,changed_input_graph_checks_verified=24,
                 independent_GPU_FP32_and_CPU_sample_checks_verified=True,
                 raw_event_values_recomputed=336,rows=rows)
    (root/'local_download_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
