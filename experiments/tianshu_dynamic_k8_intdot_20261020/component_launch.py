# SPDX-License-Identifier: Apache-2.0
"""The qualified K8V16 launch classes; only the permitted V dtype is widened."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import torch

BASE = Path(__file__).resolve().parent
LEGACY = BASE.parent / 'tianshu_k8v16_pipeline_20261019/legacy_routing.py'
POLICY = LEGACY.with_name('dispatch_policy.py')
spec = importlib.util.spec_from_file_location('_linear_v_qualified_launch_policy', POLICY)
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)
choose_config = policy.choose_config

class Producer:
    def __init__(self,kernel):self.kernel=kernel;self.tile=None;self.records={};self.decode_config=None;self.pure_decode=False
    def __getitem__(self,grid):
        def launch(*args,**kw):
            launch_grid=grid
            assert not args and kw['BLOCK_SIZE']==16 and kw['HEAD_SIZE']==128
            assert kw['num_query_heads']==16 and kw['num_queries_per_kv']==8 and kw['KV_QUANT_MODE']==0
            assert not kw['USE_TD'] and not kw['USE_TD_QO']
            assert kw['query_ptr'].dtype==torch.bfloat16 and kw['key_cache_ptr'].dtype==torch.uint8 and kw['value_cache_ptr'].dtype in (torch.bfloat16, torch.uint8)
            if kw['IS_3D']:
                n=kw['num_seqs'];selected=n in (8,16,32,40,48,56,64)
                tile=16 if selected else 32;warps=1 if selected else 2 if 8<=n<=64 else 4
                assert kw['BLOCK_M']==16 and kw['BLOCK_Q']==2
                if self.decode_config is not None:tile=self.decode_config['TILE_SIZE'];warps=self.decode_config['num_warps']
                kw.update(TILE_SIZE=tile,num_warps=warps,num_stages=1)
            elif self.pure_decode:
                kw.update(BLOCK_M=16,BLOCK_Q=2,TILE_SIZE=16,num_warps=1,num_stages=1)
                launch_grid=(kw['query_ptr'].shape[0]//2+kw['num_seqs'],grid[1])
            else:
                kw.update(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=128,num_warps=16,num_stages=1)
                launch_grid=(kw['query_ptr'].shape[0]//32+kw['num_seqs'],grid[1])
            kw['KV_CACHE_MODIFIER']='.cv';self.tile=kw['TILE_SIZE'];r=self.kernel[launch_grid](**kw)
            key=(kw['IS_3D'],kw['num_seqs'],kw['NUM_SEGMENTS_PER_SEQ'],self.tile,kw['num_warps'])
            self.records[str(key)]=dict(IS_3D=kw['IS_3D'],sequences=kw['num_seqs'],segments=kw['NUM_SEGMENTS_PER_SEQ'],tile=self.tile,grid=list(launch_grid),warps=r.metadata.num_warps,stages=r.metadata.num_stages,key_dtype=str(kw['key_cache_ptr'].dtype),query_dtype=str(kw['query_ptr'].dtype),value_dtype=str(kw['value_cache_ptr'].dtype),BLOCK_M=kw['BLOCK_M'],BLOCK_Q=kw['BLOCK_Q'],pure_decode=self.pure_decode,n_regs=getattr(r,'n_regs',None),n_spills=getattr(r,'n_spills',None),shared=getattr(r.metadata,'shared',None))
            self.last_record=self.records[str(key)]
            return r
        return launch

class Reducer:
    def __init__(self,kernel,producer):self.kernel=kernel;self.producer=producer
    def __getitem__(self,grid):
        def launch(*args,**kw):kw['TILE_SIZE']=self.producer.tile;return self.kernel[grid](*args,**kw)
        return launch


def source_proof():
    original = LEGACY.read_text()
    generated = Path(__file__).read_text()
    before = "kw['value_cache_ptr'].dtype==torch.bfloat16"
    after = "kw['value_cache_ptr'].dtype in (torch.bfloat16, torch.uint8)"
    assert generated.count(after) == 2  # one actual check and this assertion literal
    normalized = generated.replace(after, before)
    normalized = normalized.replace(",shared=getattr(r.metadata,'shared',None)", '')
    def cls(text, name):
        return next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == name)
    for name in ('Producer', 'Reducer'):
        assert ast.dump(cls(original, name)) == ast.dump(cls(normalized, name))
    return dict(legacy_path=str(LEGACY), legacy_sha256=hashlib.sha256(LEGACY.read_bytes()).hexdigest(),
                policy_path=str(POLICY), policy_sha256=hashlib.sha256(POLICY.read_bytes()).hexdigest(),
                launch_AST_unchanged_except_V_dtype_guard=True, reducer_AST_unchanged=True)


def attach(reader):
    producer = Producer(reader.kernel_unified_attention)
    reader.kernel_unified_attention = producer
    reader.reduce_segments = Reducer(reader.reduce_segments, producer)
    def invoke(kwargs):
        kw = dict(kwargs)
        assert kw['max_seqlen_q'] == 1
        n = kw['seqused_k'].numel()
        assert 1 <= n <= 64
        producer.pure_decode = True
        producer.decode_config = choose_config(n, kw['max_seqlen_k'])
        assert producer.decode_config is not None
        kw['num_par_softmax_segments'] = producer.decode_config['segments']
        return reader.unified_attention(**kw)
    return invoke, producer
