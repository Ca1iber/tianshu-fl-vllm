# SPDX-License-Identifier: Apache-2.0
"""Exercise original/extended guards, ceiling maps, configure and warmup on CPU."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS


def main():
    base=Path(__file__).resolve().parent
    original=base.parents[1]/'vllm_fl/worker/iluvatar_prefill_buckets.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest()=='966aad18aad80ef488928cafc8672cbfa2d547a832f9b151346f719ea3abc6db'
    helper=base/'extend_buckets.py'
    spec=importlib.util.spec_from_file_location('_budget_extend_cpu',helper)
    ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
    modules=[];gather=[]
    for name in ('original','extended'):
        package='_tianshu_budget_cpu_'+name
        pkg=ModuleType(package);pkg.__path__=[];sys.modules[package]=pkg
        gather_module=ModuleType(package+'.iluvatar_logits_gather')
        gather_module.enabled=lambda *a:True
        gather_module.gather_rows=lambda hidden,indices:gather.append(list(indices))
        sys.modules[gather_module.__name__]=gather_module
        spec=importlib.util.spec_from_file_location(package+'.iluvatar_prefill_buckets',original)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        modules.append(module)
    old,new=modules;provenance=ext.extend(new)
    old_helper=base.parent/'tianshu_budget4096_buckets_20261014/extend_buckets.py'
    old_spec=importlib.util.spec_from_file_location('_budget4096_reference',old_helper)
    old_ext=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old_ext);old_ext.extend(old)
    os.environ['VLLM_FL_ILUVATAR_PREFILL_BUCKETS']='1'
    config=NS(model_config=NS(hf_text_config=NS(model_type='llama',hidden_size=2048,intermediate_size=6144,num_hidden_layers=42,num_attention_heads=16,num_key_value_heads=2),quantization=None,is_multimodal_model=False,is_encoder_decoder=False),
              parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,use_ubatching=False),
              lora_config=None,speculative_config=None,scheduler_config=NS(max_num_batched_tokens=2048),
              compilation_config=NS(cudagraph_mode=NS(name='FULL_DECODE_ONLY')))
    platform=NS(vendor_name='iluvatar')
    assert old.enabled(config,platform) and new.enabled(config,platform)
    assert new.PREFILL_BUCKETS[:83]==old.PREFILL_BUCKETS
    unchanged=0;checked=0
    for budget in (1,2,4,8,256,512,1024,2048,2304,3072,4096,6144,8192):
        points=new.buckets_for_budget(budget)
        for m in range(1,budget+1):
            padded=new.padded_tokens(m,budget)
            assert m<=padded<=budget and padded==min(n for n in points if n>=m)
            if budget<=4096:
                assert old.padded_tokens(m,budget)==padded;unchanged+=1
            checked+=1
    invalid=0
    for branch,field,value in [('model_config.hf_text_config','hidden_size',4096),('model_config.hf_text_config','num_hidden_layers',40),('model_config.hf_text_config','num_key_value_heads',4),('model_config','quantization','int8'),('model_config','is_multimodal_model',True),('parallel_config','tensor_parallel_size',2),('parallel_config','use_ubatching',True),('','lora_config',object()),('','speculative_config',object())]:
        c=deepcopy(config);obj=c
        for name in branch.split('.') if branch else ():obj=getattr(obj,name)
        setattr(obj,field,value)
        assert not old.enabled(c,platform) and not new.enabled(c,platform);invalid+=1
    config.scheduler_config.max_num_batched_tokens=8192
    assert not old.enabled(config,platform) and new.enabled(config,platform)
    config.scheduler_config.max_num_batched_tokens=8193
    assert not new.enabled(config,platform)
    # Exercise the unchanged real warmup without importing Torch or initializing GPU.
    sys.modules['vllm']=ModuleType('vllm')
    sys.modules['vllm.config']=NS(CUDAGraphMode=NS(NONE='NONE'))
    sys.modules['vllm.platforms']=NS(current_platform=platform)
    sys.modules['torch']=NS(arange=lambda n,**kw:range(n),int32='int32')
    logger=NS(info=lambda *a:None)
    warmups=[]
    for module,budget in ((old,4096),(new,4096),(new,8192)):
        c=deepcopy(config);c.scheduler_config.max_num_batched_tokens=budget
        module.configure(c,platform,logger)
        points=module.buckets_for_budget(budget)
        assert c.compilation_config.cudagraph_capture_sizes==list(points)
        assert c.compilation_config.max_cudagraph_capture_size==budget
        calls=[]
        def dummy(size,**kwargs):
            assert kwargs==dict(skip_eplb=True,remove_lora=False,cudagraph_runtime_mode='NONE')
            calls.append(size);return NS(device='cpu'),None
        worker=NS(vllm_config=c,scheduler_config=c.scheduler_config,model_runner=NS(_dummy_run=dummy))
        before=len(gather);module.warmup(worker,logger)
        assert calls==list(reversed(points)) and len(gather)==before+1 and gather[-1]==list(range(64))
        warmups.append(dict(budget=budget,calls=len(calls),max=max(calls),gather_calls=1))
    data=dict(complete=True,cpu_only=True,ceiling_map_cases=checked,unchanged_low_budget_cases=unchanged,
              invalid_geometry_fallbacks=invalid,actual_original_configure_and_warmup_exercised=True,
              warmups=warmups,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (original,helper,Path(__file__))},
              extension_provenance=provenance,gpu_service_not_run=True)
    out=base/'results/20261014/prepared_32g';out.mkdir(parents=True,exist_ok=True)
    (out/'CPU_BUCKET_AUDIT_32g.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='extension_provenance'}))


if __name__=='__main__':main()
