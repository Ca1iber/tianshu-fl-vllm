# SPDX-License-Identifier: Apache-2.0
"""Bounded element offsets for the native BF16 KV writer; public slots stay int64."""
import hashlib
import importlib.util
from pathlib import Path
import re
import sys


SOURCE_SHA='f18224e51162e37c14a2e2b3045bca55baf0b207b1d203ad6ee7d8ed417139eb'


def load_variant(native,directory):
    raw=Path(native.__file__).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    text=raw.decode()
    text,n=re.subn(r': tl\.int64\b',': tl.int32',text)
    text,m=re.subn(r'\.to\(tl\.int64\)', '.to(tl.int32)',text)
    assert n and m
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    path=root/'native_kv_address32.py';path.write_text(text)
    name='_tianshu_native_kv_address32_'+hashlib.sha256(text.encode()).hexdigest()[:12]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module,dict(original_source_sha256=SOURCE_SHA,generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
                       stride_annotations_changed=n,slot_casts_changed=m,public_slot_dtype_unmodified=True,
                       floating_point_body_unmodified=True,pointer_types_unmodified=True)


class LaunchOverride:
    def __init__(self,kernel,warps,stages):
        self.kernel=kernel;self.warps=warps;self.stages=stages

    def __getitem__(self,grid):
        def invoke(*args,**kwargs):
            kwargs['num_warps']=self.warps;kwargs['num_stages']=self.stages
            return self.kernel[grid](*args,**kwargs)
        return invoke


def bound(tensor):
    return sum((n-1)*abs(s) for n,s in zip(tensor.shape,tensor.stride()))+2048*sum(abs(s) for s in tensor.stride())


def assert_safe(key,value,kcache,vcache,slots):
    bounds=[bound(t) for t in (key,value,kcache,vcache,slots)]
    assert all(0<=b<2**31 for b in bounds)
    assert kcache.shape[0]*kcache.shape[1]<2**31
    return bounds
