# SPDX-License-Identifier: Apache-2.0
"""Exact observed M64 GateUp/SiLU graph replacement; all other ranges unchanged."""
import hashlib,json,os
from pathlib import Path
import torch
from vllm.compilation.passes.inductor_pass import InductorPass,get_pass_context

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def target(n,name):return isinstance(n,torch.fx.Node) and n.op=='call_function' and str(n.target)==name
def single(n):
    assert len(n.users)==1,(n.name,list(n.users));return next(iter(n.users))
def match(node):
    if not target(node,'tianshu_prefill_int8.linear.default'):return None
    value=node.meta.get('val')
    if not isinstance(value,torch.Tensor) or tuple(value.shape)!=(64,12288) or value.dtype!=torch.bfloat16:return None
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
    assert isinstance(out.meta.get('val'),torch.Tensor) and tuple(out.meta['val'].shape)==(64,6144) and out.meta['val'].dtype==torch.bfloat16
    return dict(node=node,out=out,remove=[out,up,down,mul,sig,cast,gate,node],bank_id=node.args[4])
class FusionPass(InductorPass):
    def __init__(self,mode,directory,kernel_source):
        assert mode in ('control','candidate');self.mode=mode;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False);self.kernel_source=Path(kernel_source);self.index=0
    def uuid(self):return self.hash_dict(dict(pass_source=sha(__file__),kernel_source=sha(self.kernel_source),mode=self.mode,directory=str(self.directory),configuration=dict(bm=64,bn=64,bk=128,warps=4,stages=1,mode='two_dot')))
    def __call__(self,graph):
        context=get_pass_context().compile_range;before=graph.python_code('self').src;matches=[]
        if (context.start,context.end)==(64,64):
            matches=[m for node in list(graph.nodes) if (m:=match(node)) is not None]
            for m in matches:
                if self.mode=='candidate':
                    with graph.inserting_before(m['node']):new=graph.call_function(torch.ops.tianshu_nn_silu_fx.fused.default,args=m['node'].args)
                    new.meta=dict(m['out'].meta);m['out'].replace_all_uses_with(new)
                    for node in m['remove']:graph.erase_node(node)
            graph.lint()
        after=graph.python_code('self').src
        if self.mode=='control' or (context.start,context.end)!=(64,64):assert before==after
        row=dict(mode=self.mode,index=self.index,compile_range=[context.start,context.end],bank_ids=[m['bank_id'] for m in matches],replaced=len(matches) if self.mode=='candidate' else 0,before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest(),pass_source_sha256=sha(__file__),no_other_ranges_changed=(before==after) if (context.start,context.end)!=(64,64) else None)
        name=f'pass_{os.getpid()}_{self.index:03d}_{context.start}_{context.end}';self.index+=1
        (self.directory/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');(self.directory/(name+'.py')).write_text(after)
