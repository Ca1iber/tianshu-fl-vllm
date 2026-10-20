# SPDX-License-Identifier: Apache-2.0
"""Unchanged qualified Producer/Reducer; wrap only the 3D grid."""
import importlib.util,sys
from pathlib import Path
from source import ORDERS,remap_grid
BASE=Path(__file__).resolve().parent
NATIVE_PATH=BASE.parent/'tianshu_k8v16_address32_20261020/component_launch.py'
sys.path.insert(0,str(NATIVE_PATH.parent))
spec=importlib.util.spec_from_file_location('_cta_qualified_S16_launcher',NATIVE_PATH)
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)

class PermutedKernel:
    def __init__(self,kernel,mode):
        assert mode in ORDERS
        self.kernel,self.mode,self.last=kernel,mode,None
    def __getitem__(self,grid):
        def run(*args,**kwargs):
            is_3d=bool(kwargs['IS_3D']);actual=remap_grid(tuple(grid),self.mode,is_3d)
            compiled=self.kernel[actual](*args,**kwargs)
            self.last=dict(mode=self.mode,grid_before=list(grid),grid_after=list(actual),IS_3D=is_3d,
                grid_axis_order=list(ORDERS[self.mode]),num_warps=compiled.metadata.num_warps,
                num_stages=compiled.metadata.num_stages,kernel_hash=getattr(compiled,'hash',None))
            return compiled
        return run

def attach(reader,mode):
    wrapped=PermutedKernel(reader.kernel_unified_attention,mode)
    producer=native.Producer(wrapped)
    reader.kernel_unified_attention=producer;reader.reduce_segments=native.Reducer(reader.reduce_segments,producer)
    def invoke(kwargs):
        kw=dict(kwargs);assert kw['max_seqlen_q']==1
        n=kw['seqused_k'].numel();assert 1<=n<=64
        producer.pure_decode=True;producer.decode_config=native.choose_config(n,kw['max_seqlen_k'])
        assert producer.decode_config is not None
        kw['num_par_softmax_segments']=producer.decode_config['segments']
        answer=reader.unified_attention(**kw)
        producer.last_record['cta_order']=dict(wrapped.last)
        return answer
    return invoke,producer
