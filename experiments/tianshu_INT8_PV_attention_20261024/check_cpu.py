# SPDX-License-Identifier: Apache-2.0
"""Pure CPU/source freeze, old primitive/writer qualification binding, tail bounds."""
import ast,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import codec,launch_sweep
base=Path(__file__).resolve().parent
assert not (base/'CPU_GATE_32g.json').exists()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
primitive=json.loads(codec.PRIMITIVE_GATE.read_text());writer=json.loads(codec.WRITER_GATE.read_text())
assert primitive['complete'] and primitive['independently_CPU_rechecked']
assert primitive['frozen_sources'][str(codec.HELPER)]==sha(codec.HELPER)
for path in (codec.WRITER,codec.LAYOUT,codec.REFERENCE):
    assert writer['source_snapshots'][str(path)]==sha(path),str(path)
text,proof=codec.render_reader(codec.READER.read_bytes());compile(text,'generated_PV_reader.py','exec')
tail_cases=0
for context in (1,17,31,33,127,257,4096,5120):
    width=(context+15)//16
    for tile in (16,32,64):
        for position in range(((context+tile-1)//tile)*tile):
            if position<context:assert position//16<width
        tail_cases+=1
paths=set(base.glob('*.py'))|{base/'PLAN_32g.md',codec.READER,codec.WRITER,codec.LAYOUT,codec.REFERENCE,codec.HELPER,codec.PRIMITIVE_GATE,codec.WRITER_GATE,
    base.parent/'tianshu_soft8_mixed_planes_20261019/writer.py',
    base.parent/'tianshu_dense_w8a8_20261020/kernel.py',
    base.parent/'tianshu_V_e5m2_launch_sweep_20261024/launch_sweep.py'}
for path in paths:
    if path.suffix=='.py':ast.parse(path.read_text())
assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
gate=dict(complete=True,CPU_only=True,created_utc=datetime.now(timezone.utc).isoformat(),source_scope_proof=proof,
    existing_qualified_integer_helper_exact_SHA=True,existing_qualified_writer_layout_reference_exact_SHA=True,
    candidate_masked_tail_cases=tail_cases,declared_grid=launch_sweep.CONFIGS,
    sources={str(path):sha(path) for path in paths},GPU_component_not_run=True,model_or_service_qualified=False,adopted=False)
(base/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','source_scope_proof')},indent=2))
