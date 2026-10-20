# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=BASE.parent/'tianshu_int8_weight_nn_20261021';gate=json.loads((parent/'GPU_GATE_32g.json').read_text())
assert gate['complete'] and gate['independently_CPU_rechecked'] and all(sha(p)==v for p,v in gate['frozen_sources'].items())
assert sha(BASE/'kernel.py')==sha(parent/'kernel_nn.py')
for p in BASE.glob('*.py'):ast.parse(p.read_text())
source=(BASE/'probe_gpu.py').read_text();assert 'for m in (128,256,512,1024,2048,4096):' in source and "assert len(result['cases'])==6" in source
assert source.count('kernel.mm(q,wq.T,asc')==3 and source.count('flag_gems.linear(a.contiguous(),w,None)')==2
assert 'wq_store[:n*k].view(k,n).T' in source
assert 'for path in map(Path,frozen):' in source
assert source.count('2**24')==2 and 'assert max_sum_abs<2**24' in source
assert 2048*127*127<2**31
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'tiles.json',parent/'GPU_GATE_32g.json',BASE.parent/'tianshu_dense_w8a8_20261020/probe_gpu.py',BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json',*[Path(p) for p in gate['frozen_sources']]]
result=dict(complete=True,CPU_only=True,no_GPU_started=True,INT32_no_overflow=True,max_abs_integer_bound=2048*127*127,
    unchanged_parent_NN_kernel=True,M=[128,256,512,1024,2048,4096],tiles=len(json.loads((BASE/'tiles.json').read_text())),
    baseline='Same qualified flag_gems.linear(x.contiguous(),weight,None) route used by model prefill',model_quality_and_GPU_not_yet_proven=True,
    sources={str(p.resolve()):sha(p) for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
