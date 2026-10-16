# SPDX-License-Identifier: Apache-2.0
"""Current best operators plus an observed, guarded decode metadata path."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

BASE=Path(__file__).resolve().parents[1]
EXPERIMENTS=BASE.parent
MODE=os.environ['TIANSHU_DECODE_BATCH_META_MODE']
assert MODE in ('control','candidate')
assert os.environ['TIANSHU_TOPP_PRECISION_SERVICE_MODE']=='candidate'
assert os.environ['TIANSHU_PREFILL_GEMM_TILES_PROBE']=='1'
runpy.run_path(str(EXPERIMENTS/'tianshu_gemm_topp_geo_20261016/hooks/sitecustomize.py'))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


class BatchLoader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        render=load('_tianshu_decode_meta_render',BASE/'render.py')
        raw=Path(module.__file__).read_bytes()
        generated,derivation=render.render_source(raw)
        helper=load('_tianshu_decode_meta_kernel',BASE/'metadata.py')
        directory=Path(os.environ['TIANSHU_DECODE_BATCH_META_RECORD_DIR'])
        directory.mkdir(parents=True,exist_ok=True)
        (directory/f'generated_{os.getpid()}.py').write_text(generated)
        record=dict(pid=os.getpid(),mode=MODE,source=module.__file__,source_sha256=hashlib.sha256(raw).hexdigest(),
            derivation=derivation,calls=0,fast_calls=0,fallback_calls=0,fast_batches={},all_batches={},
            positions_dtype=None,req_indices_dtype=None,original_slot_mapping_preserved=True)
        def save():
            target=directory/f'runtime_{os.getpid()}.json'
            tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(target)
        original=module.ModelRunnerFL._prepare_inputs
        if MODE=='candidate':
            namespace=dict(vars(module),_tianshu_batch_meta=helper)
            exec(compile('from __future__ import annotations\n'+generated,str(directory/f'generated_{os.getpid()}.py'),'exec'),namespace)
            implementation=namespace['_prepare_inputs']
            launch=helper.launch
            def observed_launch(owner,n):
                record['fast_calls']+=1
                key=str(n);record['fast_batches'][key]=record['fast_batches'].get(key,0)+1
                return launch(owner,n)
            helper.launch=observed_launch
        else:implementation=original
        seen=set()
        def prepare(owner,scheduler_output,num_scheduled_tokens):
            n=owner.input_batch.num_reqs
            # Scope the experimental service to the inspected MiniCPM geometry.
            if not record['calls']:
                hf=owner.model_config.hf_config
                assert (hf.hidden_size,hf.intermediate_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads)==(2048,6144,42,16,2)
                assert owner.max_model_len==131072 and owner.use_async_scheduling and owner.speculative_config is None
            before=record['fast_calls'];result=implementation(owner,scheduler_output,num_scheduled_tokens)
            record['calls']+=1
            fast=record['fast_calls']!=before
            if not fast:record['fallback_calls']+=1
            key=str(n);record['all_batches'][key]=record['all_batches'].get(key,0)+1
            record['positions_dtype']=str(owner.positions.dtype);record['req_indices_dtype']=str(owner.req_indices.gpu.dtype)
            route=(n,fast)
            if route not in seen or record['calls']%256==0:
                seen.add(route);save()
            return result
        module.ModelRunnerFL._prepare_inputs=prepare
        save()


class BatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm_fl.worker.model_runner':return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None or spec.loader is None:raise RuntimeError('ModelRunner module missing')
        spec.loader=BatchLoader(spec.loader);return spec


sys.meta_path.insert(0,BatchFinder())
