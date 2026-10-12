# SPDX-License-Identifier: Apache-2.0
"""Compose the verified native Prefill hook with checked3D Decode launch settings."""
import importlib.abc
import importlib.machinery
import json
import hashlib
import os
from pathlib import Path
import runpy
import sys

BASE=Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE/'tianshu_native_prefill_20261012/hooks/sitecustomize.py'))


class DecodeLoader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        original=module.kernel_unified_attention;seen=set()
        source_sha=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        configuration={'TILE_SIZE':16,'num_warps':4,'num_stages':1}
        class Router:
            def __getitem__(self,grid):
                def launch(*args,**kwargs):
                    supported=(not args and kwargs.get('IS_3D') is True and
                        kwargs.get('BLOCK_M')==16 and kwargs.get('BLOCK_Q')==2 and
                        kwargs.get('num_query_heads')==16 and kwargs.get('num_queries_per_kv')==8 and
                        kwargs.get('HEAD_SIZE')==128 and kwargs.get('KV_QUANT_MODE')==0 and
                        kwargs.get('num_seqs',65)<=64)
                    if supported:kwargs.update(configuration)
                    result=original[grid](*args,**kwargs)
                    if supported:
                        key=kwargs['num_seqs']
                        if key not in seen:
                            seen.add(key)
                            root=Path(os.environ['TIANSHU_DECODE_LAUNCH_RECORD_DIR']);root.mkdir(parents=True,exist_ok=True)
                            (root/f'launch_{os.getpid()}_{key}.json').write_text(json.dumps({
                                'batch':key,'configuration':configuration,'original_grid_unchanged':True,
                                'original_kernel_body_unchanged':True,
                                'kernel_sha256':source_sha,
                                'compiled_metadata':{k:getattr(result.metadata,k,None) for k in ('num_warps','num_stages','shared')}
                                    if hasattr(result,'metadata') else None},indent=2)+'\n')
                    return result
                return launch
        module.kernel_unified_attention=Router()


class DecodeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm.v1.attention.ops.triton_unified_attention':return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        assert spec is not None and spec.loader is not None
        spec.loader=DecodeLoader(spec.loader);return spec


if os.getenv('TIANSHU_DECODE_LAUNCH_PROBE')=='1':sys.meta_path.insert(0,DecodeFinder())
