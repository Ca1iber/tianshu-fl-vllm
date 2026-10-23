# SPDX-License-Identifier: Apache-2.0
from common import *
g=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert all(sha(p)==h for p,h in g['sources'].items());logical=np.load(LOGICAL);integer=logical['xi'].astype(np.int64)@((logical['q'].astype(np.int64)-np.repeat(logical['zero'].astype(np.int64),128,axis=0))*np.repeat(logical['scale_num'].astype(np.int64),128,axis=0));expected=bf16(integer.astype(np.float32)/128);assert np.array_equal(expected,logical['reference']);rows=[]
for v in g['variants']:
 assert sha(v['file'])==v['sha256'];path=GPU/v['name']/'result.json';passed=False;different=None;phase='process_missing_result';error=None
 if path.exists():
  r=json.loads(path.read_text());assert r['CPU_gate_sha256']==sha(BASE/'CPU_GATE_32g.json') and r['source_sha256']==sha(BASE/'probe_gpu.py') and r['input_sha256']==v['sha256'];phase=r['phase'];error=r.get('error')
  if phase=='completed_numeric_pending':
   assert r['all_input_guards_versions_and_full_bytes_preserved'] and sha(r['output'])==r['output_sha256'];actual=np.fromfile(r['output'],np.uint16).reshape(64,256);different=int(np.count_nonzero(actual!=expected));passed=different==0
 rows.append(dict(variant=v['name'],passed=passed,different_cells=different,phase=phase,error=error,result_sha256=sha(path) if path.exists() else None))
result=dict(complete=True,CPU_only=True,all_16384_full_K_INT64_recomputed=True,passed_variants=[r['variant'] for r in rows if r['passed']],rows=rows,CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),performance_not_tested=True,real_model_weights_not_tested=True,adopted=False);(BASE/'FUNCTIONAL_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
