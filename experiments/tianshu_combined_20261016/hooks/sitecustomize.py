# SPDX-License-Identifier: Apache-2.0
"""Compose checked components once: currentCV,131 buckets,strictTopP16K,sampler fusion."""
import os
from pathlib import Path
import runpy
BASE = Path(__file__).resolve().parents[2]
mode = os.environ["TIANSHU_COMBO_MODE"]
assert mode in ("control", "candidate")
max_hook = (BASE / "tianshu_max_seqs_20261014/hooks/sitecustomize.py").resolve()
topp_hook = (BASE / "tianshu_decode_kv16_w2_20261014/hooks/sitecustomize.py").resolve()
budget_hook = BASE / "tianshu_budget16384_20261016/hooks/sitecustomize.py"
precision_hook = BASE / "tianshu_topp_precision_service_20261014/hooks/top_p_sitecustomize.py"
sampler_hook = BASE / "tianshu_sampler_current_20261014/hooks/sitecustomize.py"
original_run_path = runpy.run_path
base_redirected = False


def dispatch(path, *args, **kwargs):
    global base_redirected
    target = Path(path).resolve()
    if target == topp_hook:
        return original_run_path(str(precision_hook), *args, **kwargs)
    if target == max_hook and not base_redirected:
        base_redirected = True
        return original_run_path(str(budget_hook), *args, **kwargs)
    return original_run_path(path, *args, **kwargs)


runpy.run_path = dispatch
try:
    original_run_path(str(sampler_hook))
    assert base_redirected
finally:
    runpy.run_path = original_run_path
