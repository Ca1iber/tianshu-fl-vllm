# SPDX-License-Identifier: Apache-2.0
"""Install outside the unchanged qualified Top-P loader, before sampler import."""
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

FOLDER = Path(__file__).resolve().parents[1]
runpy.run_path(str(FOLDER.parent/'tianshu_gemm_topp_geo_20261016/hooks/sitecustomize.py'))


class Loader(importlib.abc.Loader):
    def __init__(self, original): self.original = original
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        geo = sys.modules['_tianshu_topp_geometric_geo']
        spec = importlib.util.spec_from_file_location('_tianshu_grid_api_route',FOLDER/'grid_route.py')
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        root = Path(os.environ['TIANSHU_TOPP_GRID_RECORD_DIR']);root.mkdir(parents=True,exist_ok=True)
        observed = dict(python_calls=0, routes={}, scope='Python launches only; not Graph replay counts')
        def observe(record):
            observed['python_calls'] += 1
            key = str((record['batch'],record['topk'],record['topp'],record['active']))
            fresh = key not in observed['routes']
            if fresh: observed['routes'][key] = dict(record,python_calls=0)
            observed['routes'][key]['python_calls'] += 1
            if fresh or observed['python_calls']%256 == 0:
                p=root/f'actual_{os.getpid()}.json';temp=p.with_suffix('.tmp')
                temp.write_text(json.dumps(observed,indent=2)+'\n');temp.replace(p)
        route = helper.install(module,geo,os.environ['TIANSHU_TOPP_GRID_CANDIDATE']=='1',observe)
        record = dict(enabled=route.enabled, native_sha256=helper.NATIVE_SHA,GEO_sha256=helper.GEO_SHA,
            route_sha256=hashlib.sha256((FOLDER/'grid_route.py').read_bytes()).hexdigest(),
            hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            original_mask_function_and_sampler_untouched=True, before_sampler_aliases=True)
        (root/f'installed_{os.getpid()}.json').write_text(json.dumps(record,indent=2)+'\n')


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname != 'vllm.v1.sample.ops.topk_topp_triton': return None
        index=sys.meta_path.index(self);sys.meta_path.pop(index)
        try:
            spec=None
            for finder in tuple(sys.meta_path):
                method=getattr(finder,'find_spec',None)
                if method is not None:
                    spec=method(fullname,path,target)
                    if spec is not None:break
        finally:sys.meta_path.insert(index,self)
        assert spec is not None and spec.loader is not None
        spec.loader=Loader(spec.loader);return spec


assert 'vllm.v1.sample.ops.topk_topp_triton' not in sys.modules
sys.meta_path.insert(0,Finder())
