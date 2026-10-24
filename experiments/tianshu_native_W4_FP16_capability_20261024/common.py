# SPDX-License-Identifier: Apache-2.0
"""Current native Down W4 pack, real activation fixtures; no GPU libraries imported."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_native_dotsaddz_down42_20261023'
spec=importlib.util.spec_from_file_location('_W4_FP16_existing_quant',PARENT/'common.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
ROOT=Path('/dev/shm/tianshu_native_W4_FP16_capability_20261024')
CPU=ROOT/'cpu'
GPU=Path('/workspace/benchmark-results/20261024_native_W4_FP16_capability_tianshu')
sha,ahash,f32=parent.sha,parent.ahash,parent.f32
DEADLINE='2026-10-24T15:58:00+00:00'

def metrics(actual,reference,kind):
    actual=np.asarray(actual,np.float64);reference=np.asarray(reference,np.float64)
    error=actual-reference
    rms=float(np.sqrt(np.mean(reference*reference)))
    relative=float(np.linalg.norm(error)/max(np.linalg.norm(reference),1.e-30))
    rtol,atol,limit=(.012,.006,.006) if kind=='BF16' else (.003,.001,.001)
    bound=rtol*abs(reference)+atol*rms
    return dict(passed=bool(np.isfinite(actual).all() and np.isfinite(reference).all() and relative<=limit and np.all(abs(error)<=bound)),
        cells=int(actual.size),failed_cells=int(np.count_nonzero(abs(error)>bound)),relative_L2=relative,
        max_abs=float(abs(error).max()),reference_RMS=rms,rtol=rtol,atol_RMS=atol,relative_L2_limit=limit)
