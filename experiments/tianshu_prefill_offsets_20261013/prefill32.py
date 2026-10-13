# SPDX-License-Identifier: Apache-2.0
"""Preserve verified no-sink L0 Prefill while comparing element-offset widths."""
import hashlib
import importlib.util
from pathlib import Path
import re
import sys


SOURCE_SHA='bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0'
L0_SHA='4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e'


def load_pair(native,directory):
    raw=Path(native.__file__).read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    text=raw.decode();old='    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)'
    assert text.count(old)==1
    text=text.replace(old,'''    if USE_SINKS:
        L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    else:
        L = tl.zeros([BLOCK_M], dtype=tl.float32)''')
    assert hashlib.sha256(text.encode()).hexdigest()==L0_SHA
    narrow,n=re.subn(r': tl\.int64\b',': tl.int32',text)
    narrow,m=re.subn(r'\.to\(\s*tl\.int64\s*\)', '.to(tl.int32)',narrow)
    assert n==28 and m==6
    root=Path(directory);root.mkdir(parents=True,exist_ok=True);modules=[];sources=[]
    for bits,code in [(64,text),(32,narrow)]:
        sha=hashlib.sha256(code.encode()).hexdigest();path=root/f'native_prefill_address{bits}.py';path.write_text(code)
        name=f'_tianshu_prefill_address{bits}_'+sha[:12]
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module);modules.append(module)
        sources.append(dict(offset_bits=bits,generated_sha256=sha,float_body_L0_preserved=True,
                            pointer_types_unmodified=True,stride_annotations_changed=n if bits==32 else 0,
                            offset_casts_changed=m if bits==32 else 0))
    return modules,sources


class PrefillKernel:
    def __init__(self,kernel):self.kernel=kernel
    def __getitem__(self,grid):
        def invoke(*args,**kwargs):
            assert not args and kwargs['IS_3D'] is False
            assert kwargs['num_queries_per_kv']==8 and kwargs['HEAD_SIZE']==128 and kwargs['KV_QUANT_MODE']==0
            block_q=32
            new_grid=(kwargs['query_ptr'].shape[0]//block_q+kwargs['num_seqs'],grid[1])
            kwargs.update(BLOCK_M=256,BLOCK_Q=block_q,TILE_SIZE=128,num_warps=16,num_stages=1)
            return self.kernel[new_grid](**kwargs)
        return invoke
