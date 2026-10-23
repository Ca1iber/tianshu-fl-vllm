# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,numpy as np
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;PARENT=EXP/'tianshu_native_W4_lm64_20261023'
spec=importlib.util.spec_from_file_location('_lm_W4_existing',PARENT/'common.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha,ahash,bf16,f32,weight,compare=old.sha,old.ahash,old.bf16,old.f32,old.weight,old.compare
MODEL,PROFILE,N,K=old.MODEL,old.PROFILE,old.N,old.K
INPUTS=old.CPU;CPU=Path('/workspace/benchmark-results/20261023_native_W4_lm_small_v2_CPU_tianshu');GPU=Path('/workspace/benchmark-results/20261023_native_W4_lm_small_v2_GPU_tianshu');MS=[1,2,4,8,16,32,48,63];VARIANTS=['BF16_actual_embedding_apply','W4_DOTS_RTN']
def rows(path,m):return np.roll(np.load(path,mmap_mode='r'),1,axis=0)[:m].copy()
