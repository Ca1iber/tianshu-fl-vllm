# SPDX-License-Identifier: Apache-2.0
import hashlib,json,py_compile,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
 assert not (BASE/'CPU_PREPARE_GATE_32g.json').exists();r=json.loads((EXP/'tianshu_native_int8_down_prefill_official_results_v2_20261022/OFFICIAL_GATE_32g.json').read_text());assert r['complete'] and r['promotion_passed'];sources=dict(r['model_frozen_sources']);assert len(sources)==890 and all(sha(p)==h for p,h in sources.items());paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run.sh',EXP/'tianshu_native_int8_lm64_20261022/kernel.py'}
 for path in BASE.glob('*.py'):py_compile.compile(str(path),doraise=True)
 sources.update({str(p):sha(p) for p in paths});gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,r2_890_sources_exact_subset=True,source_count=len(sources),source_frozen_before_GPU=True,roles=['QKV','O'],rows=[64,128,512,4096],layouts=['NN','NT'],dequant_blocks=[512,1024],sources=sources);(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
if __name__=='__main__':main()
