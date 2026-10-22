# SPDX-License-Identifier: Apache-2.0
import hashlib,json,py_compile,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (BASE/'CPU_PREPARE_GATE_32g.json').exists();r2=read(EXP/'tianshu_native_int8_down_prefill_official_results_v2_20261022/OFFICIAL_GATE_32g.json');assert r2['complete'] and r2['promotion_passed'];sources=dict(r2['model_frozen_sources']);assert len(sources)==890 and all(sha(p)==h for p,h in sources.items());assert read(EXP/'tianshu_down_W8A16_CPU_20261022/CPU_GATE_32g.json')['complete']
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run.sh',EXP/'tianshu_down_W8A16_CPU_20261022/CPU_GATE_32g.json',EXP/'tianshu_native_int8_down42_20261022/INDEPENDENT_CPU_GATE_32g.json'}
 for p in BASE.glob('*.py'):py_compile.compile(str(p),doraise=True)
 sources.update({str(p):sha(p) for p in paths});gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,r2_890_sources_exact_subset=True,source_count=len(sources),rows=[64],layouts=['NN','NT'],tiles=[[32,128],[64,64]],BK=[64,128],warps=[2,4],independent_seeds=[202610230900,202610230901],source_frozen_before_first_GPU=True,sources=sources);(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
if __name__=='__main__':main()
