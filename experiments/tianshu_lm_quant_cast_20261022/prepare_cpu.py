# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
RUNNER=Path('/workspace/benchmark-results/20261022_lm_quant_cast_runner_tianshu')
assert not RUNNER.exists();RUNNER.mkdir();(RUNNER/'cache').mkdir()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
release=json.loads((EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json').read_text());sources=dict(release['frozen_sources']);assert len(sources)==903
record=next(Path('/workspace/benchmark-results/20261022_native_int8_attention_official4k_tianshu/dense_int8_records').glob('bindings_*.json'))
rows=[r for r in json.loads(record.read_text())['launches'].values() if r['role']=='lm_head' and r['M']==64];assert len(rows)==1 and rows[0]['configuration']==dict(bm=64,bn=128,bk=128,warps=4,stages=1)
extra=[record,Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/sampler.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py')]
for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run.sh',*extra]:
 if p.suffix=='.py':ast.parse(p.read_text())
 sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items())
gate=dict(complete=True,CPU_only=True,model_not_started=True,sources=sources,source_count=len(sources),original_r3_sources=903,actual_r3_LM64=rows[0],actual_binding_source=str(record),isolated_cache=str(RUNNER/'cache'),no_service_adoption=True)
(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
