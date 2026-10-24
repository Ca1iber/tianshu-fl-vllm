# SPDX-License-Identifier: Apache-2.0
"""Independently compare every saved output to separate FP64 dtype-specific matrices."""
import subprocess
from common import *
assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());data=json.loads((GPU/'result.json').read_text())
assert data['complete'] and data['phase']=='complete' and data['CPU_gate_sha256']==sha(BASE/'CPU_GATE_32g.json')
assert data['full_inputs_versions_guards_unchanged'] and data['CPU_GPU_RNG_unchanged']
assert not data['performance_measured'] and not data['model_executed'] and not data['adopted']
assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['libraries'].items())
assert all(sha(p)==h for p,h in gate['files'].items())
expected={(kind,phase) for kind in (('BF16','FP16') if data['FP16_call_accepted'] else ('BF16',)) for phase in (0,1,2)}
assert {(row['kind'],row['phase']) for row in data['outputs']}==expected
checks=[];cells=0
for row in data['outputs']:
    kind,phase=row['kind'],row['phase'];assert sha(row['file'])==row['sha256']
    output=np.load(row['file'],allow_pickle=False);assert np.array_equal(output['actual_words'],output['eager_words'])
    values=f32(output['actual_words']) if kind=='BF16' else output['actual_words'].view(np.float16).astype(np.float32)
    reference=np.load(next(r['file'] for r in gate['references'] if r['phase']==phase),allow_pickle=False)
    check=metrics(values,reference['reference_'+kind],kind);assert check==row['metrics'] and check['passed']
    cells+=check['cells'];checks.append(dict(kind=kind,phase=phase,**check))
result=dict(complete=True,CPU_only=True,FP16_tested_contract_accepted=data['FP16_call_accepted'],FP64_cells_rechecked=cells,
    checks=checks,result_sha256=sha(GPU/'result.json'),performance_qualified=False,model_or_service_qualified=False,adopted=False)
if not data['FP16_call_accepted']:result['rejection']=data['FP16_error'];result['rejection_scope']=data['FP16_error_scope']
(BASE/'GPU_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
