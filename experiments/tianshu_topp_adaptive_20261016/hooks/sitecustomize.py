# SPDX-License-Identifier: Apache-2.0
"""Replay verified geo+GEMM profile; replace only the TopP initial cutoff."""
import os
from pathlib import Path
import runpy
BASE=Path(__file__).resolve().parents[2]
assert os.environ["TIANSHU_TOPP_PRECISION_SERVICE_MODE"]=="candidate"
assert os.environ["TIANSHU_PREFILL_GEMM_TILES_PROBE"]=="1"
assert os.environ["TIANSHU_TOPP_ADAPTIVE_MODE"] in ("control","candidate")
target=(BASE/"tianshu_gemm_topp_geo_20261016/hooks/top_p_sitecustomize.py").resolve()
original=runpy.run_path
replacements=0
def dispatch(path,*args,**kwargs):
    global replacements
    if Path(path).resolve()==target:
        replacements+=1
        return original(str(BASE/"tianshu_topp_adaptive_20261016/hooks/top_p_sitecustomize.py"),*args,**kwargs)
    return original(path,*args,**kwargs)
runpy.run_path=dispatch
try:
    original(str(BASE/"tianshu_gemm_topp_geo_20261016/hooks/sitecustomize.py"))
    assert replacements==1
finally:runpy.run_path=original
