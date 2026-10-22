# SPDX-License-Identifier: Apache-2.0
import hashlib,json,py_compile,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (BASE/'CPU_PREPARE_GATE_32g.json').exists();r3path=EXP/'tianshu_native_int8_attention_official_results_20261022/OFFICIAL_GATE_32g.json';r3=read(r3path);assert r3['complete'] and r3['promotion_passed'];sources=dict(r3['model_frozen_sources']);assert len(sources)==903 and all(sha(p)==h for p,h in sources.items());contract=EXP/'tianshu_native_weight_only_contract_CPU_results_20261022/CPU_GATE_32g.json';native=read(contract);assert native['complete'] and native['accepted_BF16_activation_cases']==12
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run.sh',r3path,contract,EXP/'tianshu_down_W8A16_CPU_20261022/CPU_GATE_32g.json',EXP/'tianshu_native_int8_down42_20261022/INDEPENDENT_CPU_GATE_32g.json'}
 providers={p:h for p,h in native['provider_sources'].items() if Path(p).suffix=='.so' or Path(p).name=='_meta_registrations.py'};assert len(providers)==3 and all(sha(p)==h for p,h in providers.items());sources.update(providers)
 for p in BASE.glob('*.py'):py_compile.compile(str(p),doraise=True)
 sources.update({str(p):sha(p) for p in paths});gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,r3_903_sources_exact_subset=True,source_count=len(sources),rows=[64],layouts=['NN','NT'],registered_op='aten::_weight_int8pack_mm',native_binaries_hashed_not_snapshotted=True,independent_seeds=[202610239200,202610239201],source_frozen_before_first_GPU=True,sources=sources);(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
if __name__=='__main__':main()
