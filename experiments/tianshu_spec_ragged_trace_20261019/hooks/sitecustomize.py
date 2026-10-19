# SPDX-License-Identifier: Apache-2.0
"""Retain the frozen failing gate; add layer observations in a separate version."""
import importlib.abc
import importlib.util
import os
from pathlib import Path
import runpy
import sys

BASE=Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE.parent/'tianshu_spec_ragged_graph_20261019/hooks/sitecustomize.py'))


class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        spec=importlib.util.spec_from_file_location('_tianshu_ragged_layer_trace',BASE/'layer_trace.py')
        observer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=observer;spec.loader.exec_module(observer)
        observer.install(module,Path(os.environ['TIANSHU_RAGGED_RECORD_DIR'])/'layer_trace')


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm_fl.worker.model_runner':return None
        position=sys.meta_path.index(self);sys.meta_path.pop(position)
        try:spec=importlib.util.find_spec(fullname)
        finally:sys.meta_path.insert(position,self)
        assert spec and spec.loader;spec.loader=Loader(spec.loader);return spec


sys.meta_path.insert(0,Finder())
