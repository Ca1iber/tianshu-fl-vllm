# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=BASE.parent/'tianshu_int8_weight_nn_20261021'
gate=json.loads((parent/'GPU_GATE_32g.json').read_text());assert gate['complete'] and gate['independently_CPU_rechecked']
assert all(sha(p)==v for p,v in gate['frozen_sources'].items())
for p in BASE.glob('*.py'):ast.parse(p.read_text())
tiles=json.loads((BASE/'tiles.json').read_text());assert len(tiles)==17 and len({tuple(c.values()) for c in tiles})==17
assert all(c['bm'] in (32,64) and c['bn'] in (128,256,512) and c['bk'] in (64,128,256) and c['warps'] in (2,4) and c['stages'] in (1,2) for c in tiles)
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'tiles.json',parent/'GPU_GATE_32g.json',*[Path(p) for p in gate['frozen_sources']]]
result=dict(complete=True,CPU_only=True,no_GPU_started=True,parent_GPU_numerical_gate_passed=True,unchanged_native_and_NN_kernel_bytes=True,finite_candidate_tiles=17,M=[8,64],source_only_not_new_GPU_proof=True,sources={str(p.resolve()):sha(p) for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
