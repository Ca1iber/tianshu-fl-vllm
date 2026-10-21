# SPDX-License-Identifier: Apache-2.0
"""Reload saved real timed outputs and independently recompute sampled attention on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True,type=Path)
args=parser.parse_args()
root=args.root.resolve()
result=json.loads((root/'result.json').read_text())
assert result['complete'] and result['phase']=='complete'
assert not torch.cuda.is_initialized()
torch.set_num_threads(8)
for filename,digest in result['runtime_sources'].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
rows=[]; sample_checks=0; full_elements=0
for case in result['cases']:
    payload=torch.load(root/f"{case['name']}_phase{case['phase']}.pt",map_location='cpu',weights_only=False)
    assert set(payload['outputs'])==set(payload['timed_outputs'])==set(case['median_ms'])
    for label,output in payload['timed_outputs'].items():
        expected=payload['expected']
        delta=(output.float()-expected.float()).abs()
        assert torch.isfinite(output).all() and not (delta>.002+.01*expected.abs()).any()
        full_elements+=output.numel()
        for sample in payload['samples']:
            qi=sample['query'].shape[0]
            for local in range(qi):
                valid=sample['length']-qi+local+1
                for head in (0,7,8,15):
                    scores=(sample['K'][:valid,head//8].float()@sample['query'][local,head].float())*(128**-.5)
                    answer=scores.softmax(0)@sample['V'][:valid,head//8].float()
                    actual=output[sample['start']+local,head].float()
                    assert torch.isfinite(actual).all() and not ((actual-answer).abs()>.002+.01*answer.abs()).any()
                    sample_checks+=1
    dense=case['median_ms'].get('dense_gather_BF16_hybrid')
    for label,median in case['median_ms'].items():
        rows.append(dict(case=case['name'],phase=case['phase'],label=label,median_ms=median,
            speedup_pct_vs_dense=None if dense is None else (dense/median-1)*100))
assert len(result['cases'])==22
summary=dict(complete=True,CPU_only_reaudit=True,source_bytes_still_match=True,
    GPU_component_only=True,model_executed=False,service_executed=False,adoption_allowed=False,
    phases=len(result['cases']),full_GPU_reference_elements_reloaded=full_elements,
    independent_CPU_head_checks=sample_checks,rows=rows,
    raw_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
    raw_tensor_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.pt'))})
base=Path(__file__).resolve().parent
(base/'GPU_GATE_32g.json').write_text(json.dumps(summary,indent=2)+'\n')
import csv
with (base/'ROUNDS_32g.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','raw_tensor_sha256')},indent=2))
