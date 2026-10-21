# SPDX-License-Identifier: Apache-2.0
"""Actual bounded gather+dense/byte fallback baseline and independent SplitKV readers."""
import hashlib,importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'tianshu_multiquery_splitkv_20261021'
PIPE=BASE.parent/'tianshu_k8v16_pipeline_20261019'
def load_py(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
source=load_py('_paths_frozen_multiq_source',OLD/'source.py');old_launch=load_py('_paths_frozen_multiq_launcher',OLD/'launch.py')
sha,segments=source.sha,source.segments
CONFIGS=[dict(block_m=m,tile=t,warps=w,stages=1) for m,t,w in ((32,32,2),(16,16,1),(32,16,2),(64,16,2))]
LABELS=['split_Q4_T32_w2','split_Q2_T16_w1','split_Q4_T16_w2','split_Q8_T16_w2']
CONTROL_CONFIG=dict(block_m=256,tile=128,warps=16,stages=1)
NATIVE=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py')
GENERATOR=BASE.parent/'tianshu_native_prefill_20261012/zero_initialization.py'
SMALL=BASE.parent/'tianshu_small_arena_20261020/policy.py'
EMULATE=BASE.parent/'tianshu_multiquery_short_diagnostic_20261021/emulate.py'
DEPENDENCIES=[source.PARENT,source.HELPERS,OLD/'source.py',OLD/'launch.py',NATIVE,GENERATOR,SMALL,PIPE/'bounded_prefill.py',PIPE/'dense_prefill.py',PIPE/'value_planes.py',EMULATE]
class DenseProducer:
    def __init__(self,kernel):self.kernel,self.last=kernel,None
    def __getitem__(self,grid):
        def call(*args,**kw):
            assert not args and not kw['IS_3D'] and kw['num_queries_per_kv']==8 and kw['HEAD_SIZE']==128
            assert str(kw['key_cache_ptr'].dtype)==str(kw['value_cache_ptr'].dtype)=='torch.bfloat16'
            c=CONTROL_CONFIG;kw.update(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=128,num_warps=16,num_stages=1)
            actual=(kw['query_ptr'].shape[0]//32+kw['num_seqs'],2);binary=self.kernel[actual](**kw)
            self.last=dict(mode='actual_gather_dense',config=c,IS_3D=False,grid=list(actual),sequences=kw['num_seqs'],query_tokens=kw['query_ptr'].shape[0],
                kernel_hash=binary.hash,n_regs=binary.n_regs,n_spills=binary.n_spills,warps=binary.metadata.num_warps,shared=binary.metadata.shared)
            return binary
        return call
class CurrentRoute:
    def __init__(self,byte_reader,byte_producer,dense_reader,dense_producer,dense):
        self.byte_reader,self.byte_producer,self.dense_reader,self.dense_producer,self.dense=byte_reader,byte_producer,dense_reader,dense_producer,dense
        self.path=None;self.active=None
    def unified_attention(self,**kw):
        # The production hybrid dense route supports causal target attention.
        # No existing 407-source draft profile is claimed for noncausal fixtures.
        staged=self.dense(kw) if kw['causal'] is True else None
        if staged is None:
            self.path='direct_byte_fallback';self.active=self.byte_producer
            return self.byte_reader.unified_attention(**kw)
        self.path='actual_bounded_gather_plus_native_BF16';self.active=self.dense_producer
        return self.dense_reader.unified_attention(**staged)
    @property
    def last(self):return dict(self.active.last,actual_route=self.path,arena_capacity_pages=self.dense.limit_pages)
class NoReducer:
    last=None

def create(root):
    import torch
    root=Path(root);models=[];generated=[]
    br,bp=source.load(root/'control_byte','control');byte_producer,unused=old_launch.attach(br,'control',None);generated.append(Path(br.__file__))
    # Load an independent native module; installed sources are never mutated.
    native=load_py('_paths_native_unified_original',NATIVE);zero=load_py('_paths_native_zero_generator',GENERATOR)
    assert not native.is_batch_invariant and not br.is_batch_invariant
    kernel,proof=zero.load_kernel(native,root/'native_dense_zero')
    dr=sys.modules['tianshu_native_zero_initialization'];dr.kernel_unified_attention=DenseProducer(kernel);dense_producer=dr.kernel_unified_attention
    generated += [Path(proof['generated_source']),root/'native_dense_zero/original_native.py']
    assert 'dense_prefill' not in sys.modules
    load_py('dense_prefill',PIPE/'dense_prefill.py')
    bounded=load_py('_paths_actual_bounded_prefill',PIPE/'bounded_prefill.py')
    small=load_py('_paths_actual_small_arena_policy',SMALL);small.install(bounded,'candidate',root/'actual_small_arena')
    dense=bounded.BoundedDensePrefill();dense.reserve('cuda');assert dense.limit_pages==8192
    current=CurrentRoute(br,byte_producer,dr,dense_producer,dense)
    models.append(dict(label='bounded_gather_or_byte',mode='current_route',config=None,reader=current,producer=current,reducer=NoReducer(),
        proof=dict(byte=bp,dense=proof,actual_small_arena_policy=True,actual_capacity_pages=8192,gather_included_in_Graph=True)))
    for label,c in zip(LABELS,CONFIGS,strict=True):
        reader,proof=source.load(root/label,'candidate');producer,reducer=old_launch.attach(reader,'candidate',c)
        models.append(dict(label=label,mode='candidate',config=c,reader=reader,producer=producer,reducer=reducer,proof=proof));generated.append(Path(reader.__file__))
    return models,generated,dense
