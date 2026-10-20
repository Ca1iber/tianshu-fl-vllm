# SPDX-License-Identifier: Apache-2.0
"""Qualified paired-K launcher; bound scratch segments independently of length."""
import importlib.util,sys
from pathlib import Path
from source import MODES,SHORT_SPLITS,configured_segments,configuration
BASE=Path(__file__).resolve().parent
NATIVE_PATH=BASE.parent/'tianshu_k8v16_address32_20261020/component_launch.py'
sys.path.insert(0,str(NATIVE_PATH.parent))
spec=importlib.util.spec_from_file_location('_short_split_qualified_launcher',NATIVE_PATH)
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)

def choose_config(mode,n,length):
    answer=native.choose_config(n,length)
    assert answer is not None
    return dict(answer,**configuration(mode,n))

class ObservedKernel:
    def __init__(self,kernel,mode):self.kernel,self.mode,self.last=kernel,mode,None
    def __getitem__(self,grid):
        def run(*args,**kwargs):
            if kwargs['IS_3D'] and kwargs['num_seqs']>24:
                assert kwargs['NUM_SEGMENTS_PER_SEQ']>=SHORT_SPLITS[self.mode]
            compiled=self.kernel[grid](*args,**kwargs)
            self.last=dict(mode=self.mode,grid=list(grid),IS_3D=bool(kwargs['IS_3D']),
                actual_tile=kwargs["TILE_SIZE"],actual_segments=kwargs["NUM_SEGMENTS_PER_SEQ"],short_active_segments=SHORT_SPLITS[self.mode],
                num_warps=compiled.metadata.num_warps,num_stages=compiled.metadata.num_stages,
                kernel_hash=getattr(compiled,'hash',None),n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None))
            return compiled
        return run

def attach(reader,mode):
    assert mode in MODES
    wrapped=ObservedKernel(reader.kernel_unified_attention,mode)
    producer=native.Producer(wrapped)
    reader.kernel_unified_attention=producer;reader.reduce_segments=native.Reducer(reader.reduce_segments,producer)
    def invoke(kwargs):
        kw=dict(kwargs);assert kw['max_seqlen_q']==1
        n=kw['seqused_k'].numel();assert 1<=n<=64
        producer.pure_decode=True;producer.decode_config=choose_config(mode,n,kw['max_seqlen_k'])
        kw['num_par_softmax_segments']=producer.decode_config['segments']
        result=reader.unified_attention(**kw)
        producer.last_record['split_policy']=dict(wrapped.last)
        return result
    return invoke,producer
