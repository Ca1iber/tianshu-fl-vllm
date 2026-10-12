# SPDX-License-Identifier: Apache-2.0
"""Explicitly enabled DSpark model/proposer compatibility, process scope only."""
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys


class DSparkLoader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        # Import while this completed module is already present in sys.modules:
        # the compatibility subclass captures the original DFlash class.
        from tianshu_dspark_adapter.proposer import DSparkProposer
        module.DFlashProposer=DSparkProposer


class DSparkFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm.v1.spec_decode.dflash':return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        assert spec is not None and spec.loader is not None
        spec.loader=DSparkLoader(spec.loader)
        return spec


if os.environ.get('TIANSHU_DSPARK_COMPAT')=='1':
    directory=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(directory))
    sys.meta_path.insert(0,DSparkFinder())
    from vllm import ModelRegistry
    ModelRegistry.register_model('Qwen3DSparkModel',
        'tianshu_dspark_adapter.model:DSparkForCausalLM')
