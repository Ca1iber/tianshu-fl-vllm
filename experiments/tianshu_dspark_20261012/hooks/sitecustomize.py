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


class VerificationAttentionLoader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        import importlib.util
        import json
        source=Path(__file__).resolve().parents[2]/'tianshu_native_prefill_20261012/zero_initialization.py'
        spec=importlib.util.spec_from_file_location('tianshu_dspark_zero_generator',source)
        generator=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=generator;spec.loader.exec_module(generator)
        record=Path(os.environ['TIANSHU_DSPARK_RECORD_DIR'])/'verification_attention'/str(os.getpid())
        candidate,provenance=generator.load_kernel(module,record)
        original=module.kernel_unified_attention
        class Router:
            def __getitem__(self,grid):
                def launch(*args,**kwargs):
                    # Preserve the original single-token 3D decode kernel.
                    selected=original if kwargs['IS_3D'] else candidate
                    launch_grid=grid
                    if (not kwargs['IS_3D'] and os.getenv('TIANSHU_DSPARK_VERIFY_TILE')=='1'
                        and kwargs['num_queries_per_kv']==8 and kwargs['HEAD_SIZE']==128
                        and kwargs['USE_CAUSAL'] and not kwargs['USE_PER_SEQ_CAUSAL']
                        and kwargs['query_ptr'].shape[0]<=8*kwargs['num_seqs']):
                        # Numerically checked for Q8, mixed Q1/Q8, B1/2/8/64.
                        # Larger prefill batches retain their original tile.
                        block_q=128//kwargs['num_queries_per_kv']
                        launch_grid=(kwargs['query_ptr'].shape[0]//block_q+kwargs['num_seqs'],grid[1])
                        kwargs.update(BLOCK_M=128,BLOCK_Q=block_q,TILE_SIZE=64,
                                      num_warps=8,num_stages=2)
                    elif (not kwargs['IS_3D'] and os.getenv('TIANSHU_DSPARK_PREFILL_TILE')=='1'
                        and kwargs['num_queries_per_kv']==8 and kwargs['HEAD_SIZE']==128
                        and kwargs['USE_CAUSAL'] and not kwargs['USE_PER_SEQ_CAUSAL']
                        and kwargs['query_ptr'].shape[0]>=16*kwargs['num_seqs']):
                        block_q=256//kwargs['num_queries_per_kv']
                        launch_grid=(kwargs['query_ptr'].shape[0]//block_q+kwargs['num_seqs'],grid[1])
                        kwargs.update(BLOCK_M=256,BLOCK_Q=block_q,TILE_SIZE=128,
                                      num_warps=16,num_stages=1)
                    return selected[launch_grid](*args,**kwargs)
                return launch
        module.kernel_unified_attention=Router()
        (record/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


class VerificationAttentionFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm.v1.attention.ops.triton_unified_attention':return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        assert spec is not None and spec.loader is not None
        spec.loader=VerificationAttentionLoader(spec.loader)
        return spec


if os.environ.get('TIANSHU_DSPARK_ZERO_VERIFY')=='1':
    sys.meta_path.insert(0,VerificationAttentionFinder())

if os.environ.get('TIANSHU_DSPARK_COMPAT')=='1':
    directory=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(directory))
    sys.meta_path.insert(0,DSparkFinder())
    from vllm import ModelRegistry
    for architecture in ('Qwen3DSparkModel','DFlashQwen3DSparkModel'):
        ModelRegistry.register_model(architecture,
            'tianshu_dspark_adapter.model:DSparkForCausalLM')
