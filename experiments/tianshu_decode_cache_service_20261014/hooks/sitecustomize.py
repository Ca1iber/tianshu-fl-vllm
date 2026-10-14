# SPDX-License-Identifier: Apache-2.0
"""Use the same composed profile, substituting only the owned Decode hook."""
from pathlib import Path
import runpy

BASE = Path(__file__).resolve().parents[2]
original_path = (BASE / "tianshu_decode_kv16_w2_20261014/hooks/decode_sitecustomize.py").resolve()
replacement_path = Path(__file__).resolve().with_name("decode_sitecustomize.py")
original_run_path = runpy.run_path


def dispatch(path, *args, **kwargs):
    if Path(path).resolve() == original_path:
        return original_run_path(str(replacement_path), *args, **kwargs)
    return original_run_path(path, *args, **kwargs)


runpy.run_path = dispatch
try:
    original_run_path(str(BASE / "tianshu_budget8192_buckets_20261014/hooks/sitecustomize.py"))
finally:
    runpy.run_path = original_run_path
