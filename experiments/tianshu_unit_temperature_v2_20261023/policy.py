# SPDX-License-Identifier: Apache-2.0
"""Remove a neutral scaling only when actual CPU request metadata proves T=1."""
import atexit,contextvars,hashlib,inspect,json,os
from pathlib import Path
STATE={};CTX=contextvars.ContextVar('tianshu_actual_unit_temperature',default=None)
def attach(metadata,values,n,all_random):
    metadata._tianshu_unit_temperature=bool(n>0 and all_random and len(values)>=n and (values[:n]==1.).all())
    metadata._tianshu_unit_temperature_tensor=metadata.temperature
    return metadata

def install(mode,root):
    import torch
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.sample.sampler import Sampler
    assert mode in ('control','candidate') and not STATE
    for cls,wanted in ((InputBatch,'de1ea4fc9d43fc6079dd375e407c2b6d93f5e1a11f49732d13654c41eaf30a65'),(Sampler,'1a2f18ddc91c89f3744ab5a93624be056bf20b7a8b2e6e13d0236182718a2253')):
        assert hashlib.sha256(Path(inspect.getsourcefile(cls)).read_bytes()).hexdigest()==wanted
    make0,sample0,temp0=InputBatch._make_sampling_metadata,Sampler.sample,Sampler.apply_temperature
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    STATE.update(mode=mode,metadata_refreshes=0,unit_metadata=0,skipped=0,unit_control=0,fallback=0,by_batch={},CPU_metadata_verified=True,policy_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():(root/f'route_{os.getpid()}.json').write_text(json.dumps(STATE,indent=2)+'\n')
    def checkpoint():
        if (STATE['skipped']+STATE['unit_control']+STATE['fallback'])%128==0:save()
    def make(owner,*a,**k):
        meta=attach(make0(owner,*a,**k),owner.temperature_cpu,owner.num_reqs,owner.all_random)
        STATE['metadata_refreshes']+=1;STATE['unit_metadata']+=int(meta._tianshu_unit_temperature);return meta
    def sample(owner,logits,metadata,*a,**k):
        token=CTX.set(metadata)
        try:return sample0(owner,logits,metadata,*a,**k)
        finally:CTX.reset(token)
    def temperature(logits,temp,all_random):
        meta=CTX.get();eligible=bool(meta is not None and getattr(meta,'_tianshu_unit_temperature',False) and getattr(meta,'_tianshu_unit_temperature_tensor',None) is temp and meta.temperature is temp and all_random and logits.is_cuda and logits.dtype==torch.float32)
        key=str(logits.shape[0]);row=STATE['by_batch'].setdefault(key,dict(skipped=0,unit_control=0,fallback=0))
        if eligible and STATE['mode']=='candidate':
            STATE['skipped']+=1;row['skipped']+=1
            checkpoint()
            return logits
        name='unit_control' if eligible else 'fallback';STATE[name]+=1;row[name]+=1
        checkpoint()
        return temp0(logits,temp,all_random)
    InputBatch._make_sampling_metadata=make;Sampler.sample=sample;Sampler.apply_temperature=staticmethod(temperature);atexit.register(save);save()
    return dict(original_temperature=temp0,save=save)
