# SPDX-License-Identifier: Apache-2.0
"""Compose validated CV/worker hooks, substituting only the owned TopP hook."""
from pathlib import Path
import runpy

BASE = Path(__file__).resolve().parents[2]
original_path = (BASE / "tianshu_decode_kv16_w2_20261014/hooks/sitecustomize.py").resolve()
replacement_path = Path(__file__).resolve().with_name("top_p_sitecustomize.py")
original_run_path = runpy.run_path


def dispatch(path, *args, **kwargs):
    if Path(path).resolve() == original_path:
        return original_run_path(str(replacement_path), *args, **kwargs)
    return original_run_path(path, *args, **kwargs)


runpy.run_path = dispatch
try:
    original_run_path(str(BASE / "tianshu_max_seqs_20261014/hooks/sitecustomize.py"))
finally:
    runpy.run_path = original_run_path
