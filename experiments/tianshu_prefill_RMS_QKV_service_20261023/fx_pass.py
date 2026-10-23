# SPDX-License-Identifier: Apache-2.0
import hashlib,json,operator,os
from pathlib import Path
import torch
from vllm.compilation.passes.inductor_pass import InductorPass,get_pass_context

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def target(node,name):return isinstance(node,torch.fx.Node) and node.op=='call_function' and str(node.target)==name

def match(node):
 if not target(node,'tianshu_native_attention.linear.default'):return None
 if not (len(node.args)==5 and isinstance(node.args[4],int) and node.args[4]%2==0):return None
 value=node.meta.get('val')
 if not isinstance(value,torch.Tensor) or tuple(value.shape)!=(4096,2560) or value.dtype!=torch.bfloat16:return None
 assert not node.kwargs and 0<=node.args[4]<=82;y=node.args[0];assert isinstance(y,torch.fx.Node) and set(y.users)=={node};residual=None
 if target(y,'vllm_ir.rms_norm.default'):
  norm=y;assert node.args[4]==0 and len(norm.args)==3 and norm.args[2]==1e-6 and not norm.kwargs;assert tuple(norm.args[0].meta['val'].shape)==(4096,2048);has=False
 else:
  assert y.op=='call_function' and y.target is operator.getitem and y.args[1]==0 and not y.kwargs;norm=y.args[0];assert target(norm,'vllm_ir.fused_add_rms_norm.default') and len(norm.args)==4 and norm.args[3]==1e-6 and not norm.kwargs;pair=list(norm.users);assert len(pair)==2 and all(n.op=='call_function' and n.target is operator.getitem and n.args[0] is norm and not n.kwargs for n in pair);residual=next(n for n in pair if n.args[1]==1);assert set(pair)=={y,residual} and node.args[4]!=0;assert tuple(y.meta['val'].shape)==tuple(residual.meta['val'].shape)==(4096,2048);has=True
 return dict(projection=node,norm=norm,y=y,residual=residual,bank_id=node.args[4],has_residual=has)

class FusionPass(InductorPass):
 def __init__(self,previous,directory):self.previous=previous;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False);self.index=0
 def uuid(self):return self.hash_dict(dict(previous=str(self.previous.uuid()),source=sha(__file__),directory=str(self.directory)))
 def __call__(self,graph):
  self.previous(graph);context=get_pass_context().compile_range;before=graph.python_code('self').src;matches=[]
  if (context.start,context.end)==(4096,4096):
   matches=[m for node in list(graph.nodes) if (m:=match(node)) is not None]
   for m in matches:
    old=m['projection'];norm=m['norm'];res=m['residual']
    with graph.inserting_before(norm):
     if m['has_residual']:
      new=graph.call_function(torch.ops.tianshu_prefill_RMS_QKV.with_residual.default,args=(*norm.args[:3],*old.args[1:]));new.meta={'val':(old.meta['val'],res.meta['val'])};out=graph.call_function(operator.getitem,args=(new,0));out.meta={'val':old.meta['val']};rr=graph.call_function(operator.getitem,args=(new,1));rr.meta={'val':res.meta['val']}
     else:
      out=graph.call_function(torch.ops.tianshu_prefill_RMS_QKV.without_residual.default,args=(*norm.args[:2],*old.args[1:]));out.meta={'val':old.meta['val']}
    old.replace_all_uses_with(out)
    if m['has_residual']:
     res.replace_all_uses_with(rr)
     for dead in (old,m['y'],res,norm):graph.erase_node(dead)
    else:
     for dead in (old,norm):graph.erase_node(dead)
   graph.lint()
  after=graph.python_code('self').src
  if (context.start,context.end)!=(4096,4096):assert before==after
  row=dict(index=self.index,compile_range=[context.start,context.end],bank_ids=[m['bank_id'] for m in matches],has_residual=[m['has_residual'] for m in matches],replaced=len(matches),before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest(),all_other_ranges_unchanged=(before==after) if (context.start,context.end)!=(4096,4096) else None,source=str(Path(__file__).resolve()),source_sha256=sha(__file__))
  name=f'pass_{os.getpid()}_{self.index:03d}_{context.start}_{context.end}';self.index+=1;(self.directory/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');(self.directory/(name+'.py')).write_text(after)
