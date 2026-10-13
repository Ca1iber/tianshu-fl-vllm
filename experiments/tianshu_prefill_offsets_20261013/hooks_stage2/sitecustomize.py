# SPDX-License-Identifier: Apache-2.0
"""Change only inspected Native L0 Prefill launch stages, preserving Decode32."""
import hashlib
import importlib.abc
import inspect
import json
import os
from pathlib import Path
import runpy
import sys


BASE=Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE/'tianshu_attention_offsets_20261013/hooks/sitecustomize.py'))


class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        path=BASE/'tianshu_native_prefill_20261012/hooks/sitecustomize.py'
        assert hashlib.sha256(path.read_bytes()).hexdigest()=='98356838abbabf53eec706a1a4a5622f2c372167b3dcccd29c2543625997800a'
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()=='5c55a8d57e438b36d176555fd9273ab9ba665dada4c57d57c19633394c2ed4e4'
        closure=inspect.getclosurevars(module.dispatch).nonlocals
        config=closure['configuration'];router=closure['router']
        assert config==dict(BLOCK_M=256,TILE_SIZE=128,num_warps=16,num_stages=1)
        assert inspect.getclosurevars(router.__getitem__).nonlocals['configuration'] is config
        config['num_stages']=2
        root=Path(os.environ['TIANSHU_PREFILL_STAGE2_RECORD_DIR']);root.mkdir(parents=True,exist_ok=True)
        (root/f'configured_{os.getpid()}.json').write_text(json.dumps(dict(
            configuration=config,base_native_hook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            same_configuration_object_used_by_kernel_router=True,decode_unchanged=True,
            kernel_body_and_grid_unchanged=True),indent=2)+'\n')


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm_fl.worker.iluvatar_hybrid_attention':return None
        for finder in tuple(sys.meta_path):
            if finder is self:continue
            method=getattr(finder,'find_spec',None);spec=method(fullname,path,target) if method else None
            if spec is not None:spec.loader=Loader(spec.loader);return spec
        return None


if os.getenv('TIANSHU_PREFILL_STAGE2_PROBE')=='1':sys.meta_path.insert(0,Finder())
