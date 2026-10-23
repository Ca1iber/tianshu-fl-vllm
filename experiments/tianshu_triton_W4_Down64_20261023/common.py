# SPDX-License-Identifier: Apache-2.0
import importlib.util,itertools,json,numpy as np
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;PARENT=EXP/'tianshu_native_dotsaddz_down42_20261023'
spec=importlib.util.spec_from_file_location('_W4_Down64_existing_reference',PARENT/'common.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha,ahash,bf16,f32,weight,hwords,compare=old.sha,old.ahash,old.bf16,old.f32,old.weight,old.hwords,old.compare
MODEL,PROFILE,LAYERS,PHASES=old.MODEL,old.PROFILE,old.LAYERS,old.PHASES
CPU=Path('/workspace/benchmark-results/20261023_triton_W4_Down64_CPU_tianshu');GPU=Path('/workspace/benchmark-results/20261023_triton_W4_Down64_GPU_tianshu')
CONFIGS=[dict(name=f'tl_m{bm}_n{bn}_k{bk}_p{int(packed)}_s{int(sym)}',bm=bm,bn=bn,bk=bk,packed=packed,symmetric=sym,warps=4,stages=1) for bm,bn,bk,packed,sym in itertools.product((32,64),(64,128),(64,128),(False,True),(False,True))]
