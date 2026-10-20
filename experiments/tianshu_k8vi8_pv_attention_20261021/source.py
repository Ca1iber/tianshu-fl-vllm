# SPDX-License-Identifier: Apache-2.0
"""Reuse the frozen GPU-proven PV primitive and generated attention variant."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_k8vi8_pv_int8_20261021'
spec=importlib.util.spec_from_file_location('_PV_attention_frozen_source',PARENT/'source.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
SOURCE,SOURCE_SHA=parent.SOURCE,parent.SOURCE_SHA
render,load_modules=parent.render,parent.load_modules
