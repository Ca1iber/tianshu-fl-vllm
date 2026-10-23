# SPDX-License-Identifier: Apache-2.0
import importlib.util,numpy as np,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;PARENT=EXP/'tianshu_native_dotsaddz_down42_20261023'
spec=importlib.util.spec_from_file_location('_dots_groups_original',PARENT/'common.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha,ahash,bf16,f32,weight,hwords,compare,pack,unpack=old.sha,old.ahash,old.bf16,old.f32,old.weight,old.hwords,old.compare,old.pack,old.unpack
MODEL,PROFILE,LAYERS,PHASES=old.MODEL,old.PROFILE,old.LAYERS,old.PHASES
CPU=Path('/workspace/benchmark-results/20261023_native_dots_groups_Down64_CPU_tianshu');GPU=Path('/workspace/benchmark-results/20261023_native_dots_groups_Down64_GPU_tianshu');GROUPS=[64,128,256]
def quant(w,group):
    wf=f32(w).reshape(2048,6144//group,group);s=bf16(np.maximum(abs(wf).max(2),np.float32(1e-8))/np.float32(7));sf=f32(s)
    q=(np.clip(np.rint(wf/sf[:,:,None]),-8,7).astype(np.int8)+8).astype(np.uint8).reshape(2048,6144).T.copy()
    return q,s.T.copy(),pack(q),bf16(-8*f32(s.T))
