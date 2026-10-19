# SPDX-License-Identifier: Apache-2.0
"""Keep native ngram stack; install validated fixed gather before profiling."""
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

BASE=Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE/'hooks/ngram_sitecustomize.py'))


class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        original=module.ModelRunnerFL.load_model
        def load_model(owner,*a,**kw):
            result=original(owner,*a,**kw)
            assert owner.speculative_config.method=='ngram_gpu' and owner.speculative_config.num_speculative_tokens==1
            source=BASE.parent/'tianshu_dspark_20261012/fixed_index.py'
            assert hashlib.sha256(source.read_bytes()).hexdigest()==FIXED_SHA
            spec=importlib.util.spec_from_file_location('_tianshu_ngram_fixed_index',source)
            fixed=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fixed;spec.loader.exec_module(fixed)
            target=importlib.import_module('flag_gems.ops.index')
            directory=Path(os.environ['TIANSHU_DSPARK_RECORD_DIR'])/'fixed_index';directory.mkdir(parents=True,exist_ok=True)
            observed=dict(python_launch_calls=0,routes={},source_sha256=FIXED_SHA,
                installed_index_source_sha256=hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest(),
                scope='Valid axis0 single1D integer indexing, other forms fall back; counts are Python launches, graph replay does not rerun observer',
                installed_before_profile_and_capture=True)
            def record(source,index,out):
                observed['python_launch_calls']+=1
                width=source.shape[1] if source.ndim==2 else 1
                key=str((source.ndim,width,str(source.dtype),str(index.dtype)))
                fresh=key not in observed['routes']
                if fresh:observed['routes'][key]=dict(source_shape=list(source.shape),index_rows=index.numel(),dtype=str(source.dtype),index_dtype=str(index.dtype),width=width)
                if fresh or observed['python_launch_calls']%512==0:
                    p=directory/f'route_{os.getpid()}.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(observed,indent=2)+'\n');tmp.replace(p)
            fixed.install(target,record)
            return result
        module.ModelRunnerFL.load_model=load_model
        graph_module=sys.modules['vllm_fl.compilation.graph']
        assert hashlib.sha256(Path(graph_module.__file__).read_bytes()).hexdigest()=='ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2'
        observer_path=BASE.parent/'tianshu_spec_graph_coverage_20261019/route_observer.py'
        assert hashlib.sha256(observer_path.read_bytes()).hexdigest()=='258d624c9f8082e82d3dea3f0d089a7d7871b58390c56a31de3fdc46726ae5e6'
        spec=importlib.util.spec_from_file_location('_tianshu_gpu_ngram_coverage',observer_path)
        observer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=observer;spec.loader.exec_module(observer)
        from vllm.forward_context import get_forward_context,is_forward_context_available
        context=lambda:get_forward_context() if is_forward_context_available() else None
        directory=Path(os.environ['TIANSHU_SPEC_COVERAGE_DIR'])
        observer.install(module.ModelRunnerFL,graph_module.GraphWrapper,context,directory,Path(os.environ['TIANSHU_COVERAGE_PHASE_FILE']))
        directory.mkdir(parents=True,exist_ok=True)
        (directory/f'installed_{os.getpid()}.json').write_text(json.dumps(dict(before_profile_capture_and_requests=True,
            native_GPU_ngram=True,model_graph_dispatch_modified=False,numeric_callbacks_installed=False,
            hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ngram_hook_sha256=hashlib.sha256((BASE/'hooks/ngram_sitecustomize.py').read_bytes()).hexdigest(),
            observer_sha256=hashlib.sha256(observer_path.read_bytes()).hexdigest()),indent=2)+'\n')



class Finder(importlib.abc.MetaPathFinder):
    resolving=False
    def find_spec(self,fullname,path=None,target=None):
        if self.resolving or fullname!='vllm_fl.worker.model_runner':return None
        self.resolving=True
        try:spec=importlib.util.find_spec(fullname)
        finally:self.resolving=False
        assert spec and spec.loader;spec.loader=Loader(spec.loader);return spec


FIXED_SHA='f48351f7ad1251acb680927d742123f92ea4f3e23505d83dfde6a3b7c713cc06'
sys.meta_path.insert(0,Finder())
