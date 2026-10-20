# SPDX-License-Identifier: Apache-2.0
import hashlib,json,math,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261020_weight_soft8_component_tianshu')
p=ROOT/'result.json'
x=json.loads(p.read_text())
assert x['complete'] and len(x['cases'])==32 and x['GPU_codecs']['complete']
for name,expected in x['frozen_sources'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    assert hashlib.sha256((ROOT/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==expected,name
full,cpu=0,0
for row in x['cases']:
    assert 'compile_error' not in row and len(row['phases'])==3
    assert set(row['median_ms'])=={'native_BF16','owned_BF16','owned_W8'}
    for phase in row['phases']:
        assert phase['RNG_unchanged'] and phase['output_partial_guards']
        for check in phase['checks'].values():
            assert check['passed'] and check['failed_elements']==check['CPU_failed_elements']==0
            actual,expected=check['CPU_actual'],check['CPU_expected']
            assert len(actual)==len(expected)
            count=0;maximum=0.0
            for arow,erow in zip(actual,expected,strict=True):
                for a,e in zip(arow,erow,strict=True):
                    assert math.isfinite(a) and math.isfinite(e)
                    delta=abs(a-e)
                    assert delta<=check['atol']+check['rtol']*abs(e)
                    maximum=max(maximum,delta);count+=1
            assert abs(maximum-check['CPU_max_abs'])<1e-12
            check.pop('CPU_actual');check.pop('CPU_expected')
            check['independently_rechecked_CPU_scalars']=count
            full+=1;cpu+=count
    for mode,raw in row['raw_ms'].items():
        assert len(raw)==7 and all(t>0 for t in raw)
        assert row['median_ms'][mode]==statistics.median(raw)
    assert abs(row['W8_speedup_percent']-100*(row['median_ms']['native_BF16']/row['median_ms']['owned_W8']-1))<1e-10
for w in x['weights'].values():assert w['saturated_elements']==0
x.update(component_only=True,adopted=False,full_GPU_FP32_checks=full,independently_rechecked_CPU_scalars=cpu,
         compiled_configurations=32,planned_configurations=32,independent_record_audit=True,
         original_result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
         analyzer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         CPU_scalar_arrays_retained_only_in_container_raw=True)
(BASE/'GEMM_GATE_32g.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(dict(complete=True,compiled_configurations=32,full_GPU_FP32_checks=full,CPU_scalars=cpu)))
