# SPDX-License-Identifier: Apache-2.0
import json
from inputs import *
assert not (BASE/'FUNCTIONAL_GATE_32g.json').exists();cpu=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert cpu['complete'] and sha(cpu['logical_input'])==cpu['logical_input_sha256'];logical=np.load(cpu['logical_input']);q=logical['q'].astype(np.int64);scale=np.repeat(logical['scale_num'].astype(np.int64),128,axis=0);zero=np.repeat(logical['zero'].astype(np.int64),128,axis=0);integer=logical['xi'].astype(np.int64)@((q-zero)*scale);expected=bf16(integer.astype(np.float32)/128);assert np.array_equal(integer,logical['integer_reference']) and np.array_equal(expected,logical['reference']);rows=[]
for name in VARIANTS:
 root=GPU/name;path=root/'result.json'
 if not path.exists():rows.append(dict(variant=name,passed=False,phase='skipped_or_process_terminated_before_result'));continue
 r=json.loads(path.read_text());assert r['CPU_gate_sha256']==sha(BASE/'CPU_GATE_32g.json') and r['source_sha256']==sha(BASE/'probe_gpu.py');proof=next(v for v in cpu['variants'] if v['name']==name);assert r['input_sha256']==proof['sha256']==sha(proof['file']);passed=False;different=None
 if r['phase']=='operations_completed_numeric_pending':
  p=Path(r['output']);assert sha(p)==r['output_sha256'] and r['all_input_storage_versions_bytes_and_guards_preserved'];actual=np.fromfile(p,dtype=np.uint16).reshape(64,256);different=int(np.count_nonzero(actual!=expected));passed=different==0
 rows.append(dict(variant=name,passed=passed,phase=r['phase'],different_BF16_cells=different,error=r.get('error'),result_sha256=sha(path)))
result=dict(complete=True,CPU_only=True,full_K_INT64_reference_recomputed=True,output_cells_per_executed_variant=16384,passed_variants=[r['variant'] for r in rows if r['passed']],rows=rows,source_sha256=sha(__file__),CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),functional_format_test_only=True,performance_not_tested=True,real_model_weights_not_tested=True,not_a_model_candidate=True);(BASE/'FUNCTIONAL_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
