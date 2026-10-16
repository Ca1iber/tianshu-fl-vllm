# SPDX-License-Identifier: Apache-2.0
"""Compose the qualified operator base with a source-verified metadata change."""
import hashlib
import importlib.abc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import textwrap

BASE=Path(__file__).resolve().parents[1]
MODE=os.environ['TIANSHU_REQUEST_POSITIONS_MODE']
assert MODE in ('control','candidate')
runpy.run_path(str(BASE.parent/'tianshu_gemm_topp_geo_20261016/hooks/sitecustomize.py'))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        renderer=load('_tianshu_positions_render',BASE/'render.py')
        kernel=load('_tianshu_positions_kernel',BASE/'positions.py')
        raw=Path(module.__file__).read_bytes()
        mode='fused' if MODE=='candidate' else 'control'
        methods,derivation=renderer.render(raw,mode)
        originals,_=renderer.render(raw,'control')
        directory=Path(os.environ['TIANSHU_REQUEST_POSITIONS_RECORD_DIR']);directory.mkdir(parents=True,exist_ok=True)
        for name in methods:
            original=getattr(module.ModelRunnerFL,name)
            assert not original.__closure__
            assert textwrap.dedent(inspect.getsource(original))==originals[name],name
            (directory/f'generated_{os.getpid()}_{name}.py').write_text(methods[name])
        record=dict(pid=os.getpid(),mode=MODE,derivation=derivation,calls=0,fused_calls=0,batches={},positions_dtype=None,requests_dtype=None,first_CPU_position_check=None)
        def save():
            p=directory/f'runtime_{os.getpid()}.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(p)
        launch=kernel.launch
        def observed_launch(owner,n):
            result=launch(owner,n)
            record['fused_calls']+=1
            return result
        kernel.launch=observed_launch
        if MODE=='candidate':
            namespace=dict(vars(module),_tianshu_positions=kernel)
            for name,text in methods.items():
                exec(compile('from __future__ import annotations\n'+text,str(directory/f'generated_{os.getpid()}_{name}.py'),'exec'),namespace)
                setattr(module.ModelRunnerFL,name,namespace[name])
        implementation=module.ModelRunnerFL._prepare_inputs
        seen=set()
        def prepare(owner,scheduler_output,counts):
            n=owner.input_batch.num_reqs
            if not record['calls']:
                hf=owner.model_config.hf_config
                assert (hf.hidden_size,hf.intermediate_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads)==(2048,6144,42,16,2)
                assert owner.max_num_reqs==256 and owner.max_model_len==131072 and owner.speculative_config is None
                assert owner.uses_mrope is False and owner.uses_xdrope_dim==0
            result=implementation(owner,scheduler_output,counts)
            record['calls']+=1
            key=str(n);record['batches'][key]=record['batches'].get(key,0)+1
            record['positions_dtype']=str(owner.positions.dtype);record['requests_dtype']=str(owner.req_indices.gpu.dtype)
            if record['first_CPU_position_check'] is None:
                total=scheduler_output.total_num_scheduled_tokens
                expected=[]
                for i,count in enumerate(counts):
                    start=int(owner.input_batch.num_computed_tokens_cpu[i]);expected.extend(range(start,start+int(count)))
                assert total==len(expected) and owner.positions[:total].cpu().tolist()==expected
                record['first_CPU_position_check']=dict(tokens=total,requests=n,passed=True)
            if n not in seen or record['calls']%512==0:
                seen.add(n);save()
            return result
        module.ModelRunnerFL._prepare_inputs=prepare
        save()


class Finder(importlib.abc.MetaPathFinder):
    resolving=False
    def find_spec(self,fullname,path=None,target=None):
        if self.resolving or fullname!='vllm_fl.worker.model_runner':return None
        # Ask the rest of the registered import chain; preserve earlier base hooks.
        self.resolving=True
        try:spec=importlib.util.find_spec(fullname)
        finally:self.resolving=False
        assert spec is not None and spec.loader is not None
        spec.loader=Loader(spec.loader)
        return spec


sys.meta_path.insert(0,Finder())
