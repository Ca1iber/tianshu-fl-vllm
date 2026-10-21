# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
assert not torch.cuda.is_initialized()
parent=BASE.parent/'tianshu_int8_nn_gateup_silu_20261021'
gate=json.loads((parent/'CPU_GATE_32g.json').read_text());assert gate['complete'];sources=dict(gate['sources'])
for p in BASE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
for p in [*BASE.glob('*.py'),BASE/'run_debug.sh',BASE/'PLAN_32g.md',parent/'CPU_GATE_32g.json']:sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
assert not torch.cuda.is_initialized()
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(dict(complete=True,CPU_only=True,CUDA_not_initialized=True,parent_kernel_unmodified=True,sources=sources),indent=2)+'\n')
print(json.dumps(dict(complete=True,CPU_only=True,parent_kernel_unmodified=True,source_count=len(sources))))
