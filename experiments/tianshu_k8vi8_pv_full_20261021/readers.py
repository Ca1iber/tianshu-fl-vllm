# SPDX-License-Identifier: Apache-2.0
"""Real current reader, native V8, and the frozen integer-PV reader."""
import hashlib,importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
from plans import PLANS,CURRENT_SHA
import source
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(root,state,current):
    root=Path(root);root.mkdir()
    def module(label,text):
        path=root/(label+'.py');path.write_text(text)
        name='_full_integer_PV_'+label;spec=importlib.util.spec_from_file_location(name,path)
        result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result);return result
    raw=source.SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest()==CURRENT_SHA
    baseline=module('current',raw.decode())
    v8text,v8proof=source.parent.render(raw);native_v8=module('native_v8',v8text)
    pv,helper,pvproof=source.load_modules(root/'pv')
    modules=dict(current=baseline,v8=native_v8,pv=pv)
    proxy=SimpleNamespace(mode=None)
    for name,reader in modules.items():
        raw_kernel,raw_reduce=reader.kernel_unified_attention,reader.reduce_segments
        class Producer:
            def __init__(self,kernel,kind):self.kernel=kernel;self.kind=kind;self.tile=None
            def __getitem__(self,grid):
                def launch(*args,**kw):
                    assert not args;label=proxy.mode;plan=PLANS[label];n=kw['num_seqs'];is3d=bool(kw['IS_3D'])
                    assert plan['reader']==self.kind and kw['KV_QUANT_MODE']==(0 if label=='K8V16' else 2)
                    assert not kw.get('USE_TD',False)
                    active=plan['pv'] and is3d
                    if self.kind=='pv':kw['PV_INT8']=active
                    launch_grid=grid
                    if is3d:kw.update(TILE_SIZE=plan['tile'] if active else 16,num_warps=plan['warps'] if active else 1,num_stages=1)
                    else:
                        kw.update(BLOCK_M=16,BLOCK_Q=2,TILE_SIZE=16,num_warps=1,num_stages=1)
                        launch_grid=(kw['query_ptr'].shape[0]//2+n,grid[1])
                    kw['KV_CACHE_MODIFIER']='.cv';self.tile=kw['TILE_SIZE'];compiled=self.kernel[launch_grid](**kw)
                    state['producer_records'][str((label,n,current[0]))]=dict(mode=label,reader=self.kind,batch=n,context=current[0],PV_INT8=active,
                        IS_3D=is3d,segments=kw['NUM_SEGMENTS_PER_SEQ'],tile=self.tile,warps=compiled.metadata.num_warps,stages=compiled.metadata.num_stages,
                        key_dtype=str(kw['key_cache_ptr'].dtype),value_dtype=str(kw['value_cache_ptr'].dtype),KV_QUANT_MODE=int(kw['KV_QUANT_MODE']),
                        n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),USE_TD=False)
                    return compiled
                return launch
        producer=Producer(raw_kernel,name)
        class Reducer:
            def __init__(self,kernel,owner):self.kernel,self.owner=kernel,owner
            def __getitem__(self,grid):
                def launch(*args,**kw):kw['TILE_SIZE']=self.owner.tile;return self.kernel[grid](*args,**kw)
                return launch
        reader.kernel_unified_attention=producer;reader.reduce_segments=Reducer(raw_reduce,producer)
    class Mux:
        __file__=pv.__file__
        def unified_attention(self,**kw):return modules[PLANS[proxy.mode]['reader']].unified_attention(**kw)
    actual=[str(Path(m.__file__)) for m in modules.values()]+[str(Path(helper.__file__))]
    return Mux(),proxy,dict(current_source_sha256=CURRENT_SHA,native_V8=v8proof,PV=pvproof,actual_sources=actual)
