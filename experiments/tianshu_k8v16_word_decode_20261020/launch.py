# SPDX-License-Identifier: Apache-2.0
"""Qualified Producer/Reducer; change K word decode or pair producer warps explicitly."""
import importlib.util,sys
from pathlib import Path
from source import MODES
BASE=Path(__file__).resolve().parent
NATIVE_PATH=BASE.parent/'tianshu_k8v16_address32_20261020/component_launch.py'
sys.path.insert(0,str(NATIVE_PATH.parent))
spec=importlib.util.spec_from_file_location('_pair_qualified_S16_launcher',NATIVE_PATH)
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)

class ObservedKernel:
    def __init__(self,kernel,mode,record_dir):
        assert mode in MODES
        self.kernel,self.mode,self.last,self.record_dir,self.saved=kernel,mode,None,Path(record_dir),set()
    def __getitem__(self,grid):
        def run(*args,**kwargs):
            if self.mode.startswith('word_'):
                key=kwargs['key_cache_ptr']
                assert key.data_ptr()%4==0 and all(s%4==0 for s in key.stride()[:-1]) and key.stride(-1)==1
            if self.mode=='pair_w2':kwargs['num_warps']=2
            compiled=self.kernel[grid](*args,**kwargs)
            self.last=dict(mode=self.mode,grid=list(grid),IS_3D=bool(kwargs['IS_3D']),
                num_warps=compiled.metadata.num_warps,num_stages=compiled.metadata.num_stages,
                kernel_hash=getattr(compiled,'hash',None),n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None))
            if kwargs['num_seqs']==64 and self.last['kernel_hash'] not in self.saved:
                import torch
                assert not torch.cuda.is_current_stream_capturing()
                self.record_dir.mkdir(parents=True,exist_ok=True)
                tag=str(self.last['kernel_hash'])
                for name in ('ttir','ttgir','llir','ptx'):
                    if name in compiled.asm:
                        value=compiled.asm[name]
                        path=self.record_dir/f'{tag}.{name}'
                        path.write_bytes(value if isinstance(value,bytes) else str(value).encode())
                self.saved.add(self.last['kernel_hash'])
            return compiled
        return run

def attach(reader,mode):
    wrapped=ObservedKernel(reader.kernel_unified_attention,mode,Path(reader.__file__).parent/'compiler_artifacts')
    producer=native.Producer(wrapped)
    reader.kernel_unified_attention=producer;reader.reduce_segments=native.Reducer(reader.reduce_segments,producer)
    def invoke(kwargs):
        kw=dict(kwargs);assert kw['max_seqlen_q']==1
        n=kw['seqused_k'].numel();assert 1<=n<=64
        producer.pure_decode=True;producer.decode_config=native.choose_config(n,kw['max_seqlen_k'])
        assert producer.decode_config is not None
        kw['num_par_softmax_segments']=producer.decode_config['segments']
        answer=reader.unified_attention(**kw)
        producer.last_record['K_load_variant']=dict(wrapped.last)
        return answer
    return invoke,producer
