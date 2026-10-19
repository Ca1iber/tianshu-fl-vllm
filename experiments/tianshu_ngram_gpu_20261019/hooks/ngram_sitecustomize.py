# SPDX-License-Identifier: Apache-2.0
"""Native ngram, supported precision rejection, and verified Q1/Q2 tiles."""
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

BASE=Path(__file__).resolve().parents[2]/'tianshu_ngram_framework_20261016'
runpy.run_path(str(BASE.parent/'tianshu_gemm_topp_geo_20261016/hooks/sitecustomize.py'))
assert os.environ['TIANSHU_NGRAM_MODE']=='candidate'
DIRECTORY=Path(os.environ['TIANSHU_DSPARK_RECORD_DIR'])


def write(kind,data):
    DIRECTORY.mkdir(parents=True,exist_ok=True);path=DIRECTORY/f'{kind}_{os.getpid()}.json'
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        if module.__name__=='vllm.v1.sample.rejection_sampler':
            sha=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            assert sha=='953bb182ac2e0ef8a2c07b9ac3e157e3542f3159b3ae6f3f3f5378acfe851a86'
            compat=load('_tianshu_ngram_rejection_compat',BASE.parent/'tianshu_dspark_20261012/rejection_compat.py')
            assert hashlib.sha256(Path(compat.__file__).read_bytes()).hexdigest()==COMPAT_SHA
            compat.install(module)
            original=module.RejectionSampler.__init__
            def initialize(owner,sampler,spec_config=None,device=None):
                original(owner,sampler,spec_config,device)
                assert spec_config.method=='ngram_gpu' and spec_config.num_speculative_tokens==1
                assert spec_config.rejection_sample_method=='standard'
                assert owner.use_fp64_gumbel is False and owner.synthetic_mode is False
                write('rejection',dict(source_sha256=sha,use_fp64_gumbel=False,synthetic_mode=False,
                    standard_acceptance_ratio_unchanged=True,positive_FP32_uniform=True))
            module.RejectionSampler.__init__=initialize
        else:
            assert module.__name__=='vllm_fl.worker.model_runner'
            assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()=='b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'
            original=module.ModelRunnerFL.load_model
            def load_model(owner,*args,**kwargs):
                result=original(owner,*args,**kwargs)
                spec=owner.speculative_config;hf=owner.model_config.hf_config
                assert spec.method=='ngram_gpu' and spec.num_speculative_tokens==1 and (spec.prompt_lookup_min,spec.prompt_lookup_max)==(3,5)
                assert (hf.hidden_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads)==(2048,42,16,2)
                assert owner.max_model_len==131072 and not owner.model_config.use_fp64_gumbel
                assert owner.use_async_scheduling and owner.use_async_spec_decode, 'GPU ngram should preserve native asynchronous scheduling'
                import vllm.v1.attention.ops.triton_unified_attention as native
                import vllm_fl.worker.iluvatar_prefill_buckets as buckets
                from vllm.platforms import current_platform
                assert buckets.enabled(owner.vllm_config,current_platform) is False
                import vllm.v1.attention.backends.triton_attn as backend
                import vllm_fl.worker.iluvatar_hybrid_attention as hybrid
                assert hybrid._INSTALLED is False
                verify=load('_tianshu_ngram_native_verify',BASE/'native_verify.py')
                routes=[]
                def record(row):routes.append(row);write('verification_routes',dict(routes=routes))
                owner._tianshu_verify_dispatch=verify.install(native,sys.modules['tianshu_native_cache_policy'],backend,
                    DIRECTORY/'split_generated',enable_split=True,record=record)
                proposer=owner.drafter;original_propose=proposer.propose
                counts=dict(calls=0,requests_seen=0,host_synchronization_added=False)
                def propose(*a,**kw):
                    proposed=original_propose(*a,**kw);counts['calls']+=1
                    assert isinstance(proposed,tuple) and len(proposed)==2
                    counts['requests_seen']+=proposed[0].shape[0]
                    counts['last_draft_shape']=list(proposed[0].shape)
                    if counts['calls']==1 or counts['calls']%512==0:write('proposer',counts)
                    return proposed
                proposer.propose=propose
                write('model',dict(method=spec.method,num_speculative_tokens=spec.num_speculative_tokens,
                    prompt_lookup_min=spec.prompt_lookup_min,prompt_lookup_max=spec.prompt_lookup_max,
                    async_scheduling=owner.use_async_scheduling,async_spec_decode=owner.use_async_spec_decode,
                    prefill75_enabled=False,hybrid_installed=False,small_attention_installed_before_profile_and_capture=True,
                    native_GPU_proposer_source_sha256=hashlib.sha256(Path(sys.modules[proposer.__module__].__file__).read_bytes()).hexdigest()))
                return result
            module.ModelRunnerFL.load_model=load_model


class Finder(importlib.abc.MetaPathFinder):
    resolving=False
    def find_spec(self,fullname,path=None,target=None):
        if self.resolving or fullname not in ('vllm.v1.sample.rejection_sampler','vllm_fl.worker.model_runner'):return None
        self.resolving=True
        try:spec=importlib.util.find_spec(fullname)
        finally:self.resolving=False
        assert spec and spec.loader;spec.loader=Loader(spec.loader);return spec


COMPAT_SHA='80df34a4e9e5fb74d46f86698d7c50d68212d7fe94d0941fc32a893fb40bf415'
sys.meta_path.insert(0,Finder())
