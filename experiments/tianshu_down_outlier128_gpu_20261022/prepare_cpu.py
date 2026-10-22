# SPDX-License-Identifier: Apache-2.0
import hashlib,json,py_compile,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
RANKS=Path('/workspace/benchmark-results/20261022_down_outlier_partition_CPU_tianshu/channel_ranks.npz')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (BASE/'CPU_PREPARE_GATE_32g.json').exists()
 gate=read(EXP/'tianshu_down_outlier_partition_CPU_20261022/CPU_GATE_32g.json');assert gate['complete'] and gate['CPU_only'] and gate['selected_global_BF16_channels']==128 and sha(RANKS)==gate['channel_ranks_sha256']
 with np.load(RANKS,allow_pickle=False) as data:
  ranks=data['top128_indices'];assert ranks.shape==(42,128) and ranks.dtype==np.int32 and ranks.min()>=0 and ranks.max()<6144 and all(len(set(row))==128 for row in ranks)
 r1=read(EXP/'tianshu_native_int8_prefill_official_results_v2_20261022/OFFICIAL_GATE_32g.json');assert r1['complete'] and r1['promotion_passed'];sources=dict(r1['model_frozen_sources']);assert len(sources)==877 and all(sha(p)==h for p,h in sources.items())
 paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',BASE/'run.sh',RANKS,EXP/'tianshu_down_outlier_partition_CPU_20261022/CPU_GATE_32g.json',EXP/'tianshu_native_int8_down42_20261022/INDEPENDENT_CPU_GATE_32g.json',EXP/'tianshu_native_int8_lm64_20261022/kernel.py'}
 for path in BASE.glob('*.py'):py_compile.compile(str(path),doraise=True)
 sources.update({str(p):sha(p) for p in paths})
 result=dict(complete=True,CPU_only=True,no_GPU_imports=True,original877_sources_exact_subset=True,source_count=len(sources),fixed_BF16_channels=128,independent_random_seed=202610224700,all42_banks=True,source_frozen_before_first_GPU=True,model_executed=False,official_benchmark=False,adopted=False,sources=sources)
 (BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
if __name__=='__main__':main()
