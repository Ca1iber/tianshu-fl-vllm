# SPDX-License-Identifier: Apache-2.0
import hashlib,json,struct
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261023_native_INT4_smoke_GPU_tianshu')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (BASE/'FUNCTIONAL_GATE_32g.json').exists();r=json.loads((ROOT/'result.json').read_text());cpu=json.loads((BASE/'CPU_PREPARE_32g.json').read_text());assert r['complete'] and r['CPU_prepare_sha256']==sha(BASE/'CPU_PREPARE_32g.json') and r['source_sha256']==sha(BASE/'smoke_gpu.py') and r['functional_smoke_only'] and r['performance_not_tested'] and r['real_model_weights_not_tested'] and r['checkpoint_not_opened']
 for p,h in cpu['sources'].items():assert sha(p)==h
 passed=r['phase']=='operations_completed_numeric_pending' and r['CUDA_operations_completed'] and len(r['rows'])==4;cells=0
 for row in r['rows']:
  p=Path(row['raw']);assert sha(p)==row['raw_sha256'] and p.stat().st_size==row['raw_bytes']==64*256*2;raw=p.read_bytes();expected=256*.125*(row['nibble']-8)*.125;assert expected==row['expected_scalar'];actual=[struct.unpack('<f',b'\0\0'+raw[i:i+2])[0] for i in range(0,len(raw),2)];row_pass=all(value==expected for value in actual);passed=passed and row_pass;cells+=len(actual)
 gate=dict(complete=True,CPU_only=True,functionality_passed=passed,checked_output_cells=cells,CUDA_operations_completed=r['CUDA_operations_completed'],GPU_phase=r['phase'],error=r.get('error'),predeclared_symmetric_formula_not_changed=True,performance_not_tested=True,real_model_weights_not_tested=True,not_a_model_candidate=True,raw_result_sha256=sha(ROOT/'result.json'),verifier_sha256=sha(__file__),CPU_prepare_sha256=sha(BASE/'CPU_PREPARE_32g.json'));(BASE/'FUNCTIONAL_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
