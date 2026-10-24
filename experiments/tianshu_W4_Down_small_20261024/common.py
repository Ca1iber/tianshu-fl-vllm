# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
ROOT=Path('/dev/shm/tianshu_W4_Down_small_20261024');CPU=ROOT/'cpu';GPU=ROOT/'gpu'
PARENT=EXP/'tianshu_native_dotsaddz_down42_20261023'
spec=importlib.util.spec_from_file_location('_Down_small_exact_existing_CPU',PARENT/'common.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha,ahash,bf16,f32,weight,hwords,compare=old.sha,old.ahash,old.bf16,old.f32,old.weight,old.hwords,old.compare
MODEL=old.MODEL;MS=[1,2,4,8,16,32,48,63];LAYERS=[0,20,41];PHASES=[0,1,2]
VARIANTS=['BF16_actual_parent','W4_direct_or_pad32_64']
DEADLINE='2026-10-24T15:58:00+00:00'
