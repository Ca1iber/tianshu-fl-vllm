# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only Python thread snapshots while holding the interpreter GIL."""
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

BASE = Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE / 'tianshu_buckets75_recheck_20261012/joint_hooks/sitecustomize.py'))

if os.environ.get('TIANSHU_CPU_STACK_DIR'):
    root = Path(os.environ['TIANSHU_CPU_STACK_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    _stack_file = root / f'stacks_{os.getpid()}.jsonl'

    def collect():
        with _stack_file.open('w', buffering=1) as output:
            while True:
                time.sleep(1)
                frames = sys._current_frames()
                stacks = {}
                for thread_id, frame in frames.items():
                    rows = []
                    current = frame
                    while current is not None and len(rows) < 80:
                        code = current.f_code
                        rows.append([code.co_filename, code.co_name, current.f_lineno])
                        current = current.f_back
                    stacks[str(thread_id)] = rows
                del frames, frame, current
                output.write(json.dumps(dict(wall_time=time.time(),
                                             monotonic=time.monotonic(), stacks=stacks)) + '\n')

    threading.Thread(target=collect, daemon=True, name='tianshu_python_stack_sampler').start()
