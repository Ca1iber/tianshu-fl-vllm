# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;RUNNER=Path('/workspace/benchmark-results/20261022_LM_pad64_runner_tianshu');assert not RUNNER.exists();RUNNER.mkdir();(RUNNER/'cache').mkdir()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
sources=dict(json.loads((EXP/'tianshu_native_int8_attention_prefill_release_20261022/serve_profile_template.json').read_text())['frozen_sources']);assert len(sources)==903
route=json.loads(Path('/workspace/benchmark-results/20261022_attention_BF16_route_GPU_tianshu/result.json').read_text());linear=Path(route['flag_gems_linear_source']);assert sha(linear)==route['flag_gems_linear_sha256']
for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run.sh',linear,Path('/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/vocab_parallel_embedding.py')]:
 if p.suffix=='.py':ast.parse(p.read_text())
 sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items());gate=dict(complete=True,CPU_only=True,sources=sources,source_count=len(sources),original903_sources=True,actual_vendor_linear_source=str(linear),actual_vendor_linear_sha256=sha(linear),M_points=[16,32,48,63,64],cutoff_choices=[32,48,64],isolated_cache=str(RUNNER/'cache'));(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
