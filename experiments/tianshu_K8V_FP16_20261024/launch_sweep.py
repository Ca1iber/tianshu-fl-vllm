# SPDX-License-Identifier: Apache-2.0
"""Native FP16 V launch comparison; compiled-code recording occurs before timed replay."""
import hashlib
import json
from pathlib import Path

CONFIGS = [dict(label=f"nativeF16_S4_T{t}_W1", segments=4, TILE_SIZE=t, num_warps=1)
           for t in (16,32)]

class CompiledRecorder:
    def __init__(self, wrapped, root, label, target):
        self.wrapped=wrapped; self.root=Path(root)/label; self.target=target
        self.seen=set(); self.records=[]
    def __getattr__(self,name): return getattr(self.wrapped,name)
    def __getitem__(self,grid):
        launch=self.wrapped[grid]
        def invoke(*args,**kwargs):
            kernel=launch(*args,**kwargs)
            key=str(getattr(kernel,'hash',getattr(kernel,'name','unknown')))
            if key not in self.seen:
                self.seen.add(key)
                out=self.root/hashlib.sha256(key.encode()).hexdigest()[:16]
                out.mkdir(parents=True,exist_ok=False)
                files={}
                for kind in ('ttir','ttgir','llir','asm'):
                    body=kernel.asm.get(kind)
                    if isinstance(body,str):
                        path=out/('kernel_unified_attention.'+kind)
                        path.write_text(body)
                        files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                assert 'asm' in kernel.asm and files
                record=dict(target=self.target,compiled_key=key,files=files,
                    n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None),
                    shared=getattr(kernel.metadata,'shared',None),num_warps=kernel.metadata.num_warps,
                    num_stages=kernel.metadata.num_stages,
                    recorded_before_timed_Graph_replay=True)
                (out/'metadata.json').write_text(json.dumps(record,indent=2)+'\n')
                self.records.append(record)
            return kernel
        return invoke

def attach(model,component_launch,output):
    import torch
    import triton
    ordinary,producer=component_launch.attach(model['reader'])
    target=triton.runtime.driver.active.get_current_target()
    target_record=dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size)
    recorder=CompiledRecorder(producer.kernel,output,model['label'],target_record)
    producer.kernel=recorder
    producer._sweep_codegen=recorder
    if model['label']=='control': return ordinary,producer
    config=dict(model['config'])
    assert config in CONFIGS
    def invoke(kwargs):
        kw=dict(kwargs)
        assert kw['q'].dtype==torch.bfloat16 and kw['k'].dtype==torch.uint8 and kw['v'].dtype==torch.float16
        assert kw['k'].stride(-1)==kw['v'].stride(-1)==1
        assert kw['max_seqlen_q']==1 and kw['seqused_k'].numel()==64
        producer.pure_decode=True
        producer.decode_config={k:config[k] for k in ('segments','TILE_SIZE','num_warps')}
        kw['num_par_softmax_segments']=config['segments']
        answer=model['reader'].unified_attention(**kw)
        assert producer.last_record['IS_3D']
        assert producer.last_record['segments']==config['segments']
        assert producer.last_record['tile']==config['TILE_SIZE']
        assert producer.last_record['warps']==config['num_warps']
        producer.last_record['address_selection']=dict(recorder.last_selection)
        return answer
    return invoke,producer
