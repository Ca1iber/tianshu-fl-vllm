# SPDX-License-Identifier: Apache-2.0
"""Reuse independently validated generic integer primitive; enumerate only new M."""
import ast,hashlib,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parent=BASE.parent/'tianshu_dense_w8a8_20261020'
    old=json.loads((parent/'CPU_GATE_32g.json').read_text());assert old['complete'] and old['INT32_no_overflow']
    assert all(sha(p)==v for p,v in old['sources'].items())
    assert sha(BASE/'kernel.py')==sha(parent/'kernel.py')=='b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152'
    spec=importlib.util.spec_from_file_location('new_batch_probe_CPU_enumeration',BASE/'probe_gpu.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    assert probe.SHAPES==[('gateup',12288,2048),('lm_head',130560,2048)] and len(probe.PLANS)==4
    assert 2048*127*127<2**31
    for path in BASE.glob('*.py'):ast.parse(path.read_text())
    sources={str(p):sha(p) for p in [*BASE.glob('*.py'),*BASE.glob('*.sh'),parent/'CPU_GATE_32g.json']}
    result=dict(complete=True,no_GPU_started=True,INT32_no_overflow=True,max_absolute_integer_accumulation=2048*127*127,
        primitive_byte_identical_to_qualified_kernel=True,generic_CPU_gate_rechecked=True,actual_new_M_GPU_checks_pending=True,
        M=[32,37,40,48,56,63],roles=['gateup','lm_head'],configuration_count=4,case_count=12,sources=sources)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
if __name__=='__main__':main()
