# SPDX-License-Identifier: Apache-2.0
"""Retain current CV profile and record actual WorkerFL scheduling configuration."""
from pathlib import Path
import runpy
BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / "tianshu_max_seqs_20261014/hooks/sitecustomize.py"))
