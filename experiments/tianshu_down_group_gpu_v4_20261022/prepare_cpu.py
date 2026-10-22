# SPDX-License-Identifier: Apache-2.0
import hashlib,json,py_compile,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (BASE/'CPU_PREPARE_GATE_32g.json').exists();r1=read(EXP/'tianshu_native_int8_prefill_official_results_v2_20261022/OFFICIAL_GATE_32g.json');assert r1['complete'] and r1['promotion_passed'];sources=dict(r1['model_frozen_sources']);assert len(sources)==877 and all(sha(p)==h for p,h in sources.items());cpu=read(EXP/'tianshu_down_group_activation_CPU_20261022/CPU_GATE_32g.json');assert cpu['complete'] and cpu['baseline_codes_scales_INT32_BF16_exact']
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run.sh',EXP/'tianshu_down_group_activation_CPU_20261022/CPU_GATE_32g.json',EXP/'tianshu_native_int8_down42_20261022/INDEPENDENT_CPU_GATE_32g.json',EXP/'tianshu_native_int8_lm64_20261022/kernel.py'}
 for p in BASE.glob('*.py'):py_compile.compile(str(p),doraise=True)
 sources.update({str(p):sha(p) for p in paths});gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,original877_sources_exact_subset=True,source_count=len(sources),groups=[128,512],warps=[2,4],independent_seed=202610228900,single_bank_cache_screen_only=True,layouts=['NT','NN'],tiles=[[32,128],[64,64]],rows=[64],source_frozen_before_first_GPU=True,model_executed=False,official_benchmark=False,adopted=False,sources=sources);(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
if __name__=='__main__':main()
