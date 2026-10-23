# SPDX-License-Identifier: Apache-2.0
import importlib.util,json,numpy as np
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;PARENT=EXP/'tianshu_native_W4_lm_small_v2_20261023';FULL=EXP/'tianshu_native_W4_lm64_20261023'
spec=importlib.util.spec_from_file_location('_LM_pad_existing',PARENT/'common.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha,ahash,bf16,f32,rows,MODEL,PROFILE,INPUTS,N,K=old.sha,old.ahash,old.bf16,old.f32,old.rows,old.MODEL,old.PROFILE,old.INPUTS,old.N,old.K
CPU=Path('/workspace/benchmark-results/20261023_W4_LM_pad32_64_v2_CPU_tianshu');GPU=Path('/workspace/benchmark-results/20261023_W4_LM_pad32_64_v2_GPU_tianshu');TIMING_MS=[1,2,4,8,16,32,48,63]
def bucket(m):return 32 if m<=32 else 64
