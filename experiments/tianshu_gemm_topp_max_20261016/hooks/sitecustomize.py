# SPDX-License-Identifier: Apache-2.0
"""Current CV/default scheduler plus strictTopP32Kmax and guarded M2048 GEMM."""
import os
from pathlib import Path
import runpy
BASE=Path(__file__).resolve().parents[2]
mode=os.environ['TIANSHU_TOPP_PRECISION_SERVICE_MODE']
assert mode in ('control','candidate')
assert os.environ['TIANSHU_PREFILL_GEMM_TILES_PROBE']==('1' if mode=='candidate' else '0')
runpy.run_path(str(BASE/'tianshu_gemm_topp_max_20261016/hooks/compose_top_p.py'))
# Register the existing GEMM finder without composing its old base twice.
sql_hook=(BASE/'tianshu_sql_workflow_20261014/hooks/sitecustomize.py').resolve()
original=runpy.run_path
skipped=0

def dispatch(path,*args,**kwargs):
    global skipped
    if Path(path).resolve()==sql_hook:
        skipped+=1
        return {}
    return original(path,*args,**kwargs)

runpy.run_path=dispatch
try:
    original(str(BASE/'tianshu_prefill_gemm_tiles_20261014/hooks/sitecustomize.py'))
    assert skipped==1
finally:
    runpy.run_path=original
