# SPDX-License-Identifier: Apache-2.0
"""Keep the current CV profile; replace only the maximum-budget bucket helper."""
from pathlib import Path
import runpy
BASE = Path(__file__).resolve().parents[2]
OLD = (BASE / "tianshu_budget8192_buckets_20261014/extend_buckets.py").resolve()
NEW = Path(__file__).resolve().parents[1] / "extend_buckets.py"
original = runpy.run_path

def dispatch(path, *args, **kwargs):
    return original(str(NEW) if Path(path).resolve() == OLD else path, *args, **kwargs)

runpy.run_path = dispatch
try:
    original(str(BASE / "tianshu_max_seqs_20261014/hooks/sitecustomize.py"))
finally:
    runpy.run_path = original
