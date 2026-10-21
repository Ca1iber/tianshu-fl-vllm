# SPDX-License-Identifier: Apache-2.0
"""Exact observed M64 GateUp/SiLU graph replacement; all other ranges unchanged."""
import hashlib,json,os
from pathlib import Path
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def target(n,name):return isinstance(n,torch.fx.Node) and n.op=='call_function' and str(n.target)==name
def single(n):
    assert len(n.users)==1,(n.name,list(n.users));return next(iter(n.users))
def match(node):
    if not target(node,'tianshu_prefill_int8.linear.default'):return None
    value=node.meta.get('val')
    if not isinstance(value,torch.Tensor) or (value.ndim!=2 or value.shape[1]!=12288) or value.dtype!=torch.bfloat16:return None
    assert not node.kwargs and len(node.args)==5 and isinstance(node.args[4],int) and 0<=node.args[4]<42
    assert len(node.users)==2
    slices=list(node.users);assert all(target(n,'aten.slice.Tensor') and not n.kwargs and len(n.args)==4 and n.args[:2]==(node,1) for n in slices)
    gate=next(n for n in slices if n.args[2:]==(0,6144));up=next(n for n in slices if n.args[2:]==(6144,9223372036854775807))
    cast=single(gate);assert target(cast,'prims.convert_element_type.default') and cast.args==(gate,torch.float32) and not cast.kwargs
    assert len(cast.users)==2
    sig=next(n for n in cast.users if target(n,'aten.sigmoid.default'));assert sig.args==(cast,) and not sig.kwargs
    mul=single(sig);assert target(mul,'aten.mul.Tensor') and mul.args==(cast,sig) and set(cast.users)=={sig,mul} and not mul.kwargs
    down=single(mul);assert target(down,'prims.convert_element_type.default') and down.args==(mul,torch.bfloat16) and not down.kwargs
    out=single(down);assert target(out,'aten.mul.Tensor') and out.args==(down,up) and single(up) is out and not out.kwargs
    assert isinstance(out.meta.get('val'),torch.Tensor) and out.meta['val'].ndim==2 and out.meta['val'].shape[1]==6144 and str(out.meta['val'].shape[0])==str(value.shape[0]) and out.meta['val'].dtype==torch.bfloat16
    return dict(node=node,out=out,remove=[out,up,down,mul,sig,cast,gate,node],bank_id=node.args[4])
