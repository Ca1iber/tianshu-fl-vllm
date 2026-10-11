# SPDX-License-Identifier: Apache-2.0
"""Compose the exact tested tile hook with the tested gather module hook."""
from pathlib import Path
import runpy

base=Path(__file__).resolve().parents[2]
runpy.run_path(str(base/'tianshu_fa2_tiles_20261012/hooks/sitecustomize.py'))
runpy.run_path(str(base/'tianshu_gather_scope_20261012/hooks/sitecustomize.py'))
