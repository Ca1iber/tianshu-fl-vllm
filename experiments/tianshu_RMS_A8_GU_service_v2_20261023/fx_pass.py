# SPDX-License-Identifier: Apache-2.0
import hashlib,json,operator,os
from pathlib import Path
import torch
from vllm.compilation.passes.inductor_pass import InductorPass,get_pass_context

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def target(n,name):return isinstance(n,torch.fx.Node) and n.op=='call_function' and str(n.target)==name

def match(node):
 if not target(node,'tianshu_nn_silu_fx.fused.default'):return None
 value=node.meta.get('val')
 if not isinstance(value,torch.Tensor) or tuple(value.shape)!=(64,6144) or value.dtype!=torch.bfloat16:return None
 assert not node.kwargs and len(node.args)==5 and isinstance(node.args[4],int) and 0<=node.args[4]<42
 y=node.args[0];assert y.op=='call_function' and y.target is operator.getitem and y.args[1]==0 and not y.kwargs and set(y.users)=={node}
 norm=y.args[0];assert target(norm,'vllm_ir.fused_add_rms_norm.default') and len(norm.args)==4 and norm.args[3]==1.e-6 and not norm.kwargs
 pair=list(norm.users);assert len(pair)==2 and all(n.op=='call_function' and n.target is operator.getitem and n.args[0] is norm and not n.kwargs for n in pair)
 residual=next(n for n in pair if n.args[1]==1);assert set(pair)=={y,residual}
 assert tuple(y.meta['val'].shape)==tuple(residual.meta['val'].shape)==(64,2048)
 return dict(consumer=node,norm=norm,y=y,residual=residual,bank_id=node.args[4])

class FusionPass(InductorPass):
 def __init__(self,previous,mode,directory):
  self.previous=previous;self.mode=mode;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False);self.index=0
 def uuid(self):return self.hash_dict(dict(previous=str(self.previous.uuid()),source=sha(__file__),mode=self.mode,directory=str(self.directory)))
 def __call__(self,graph):
  self.previous(graph);context=get_pass_context().compile_range;before=graph.python_code('self').src;matches=[]
  if (context.start,context.end)==(64,64):
   matches=[m for node in list(graph.nodes) if (m:=match(node)) is not None]
   if self.mode=='candidate':
    for m in matches:
     old=m['consumer'];norm=m['norm'];res=m['residual']
     args=(*norm.args[:3],*old.args[1:])
     with graph.inserting_before(norm):
      new=graph.call_function(torch.ops.tianshu_RMS_A8_GU_v2.fused.default,args=args);new.meta={'val':(old.meta['val'],res.meta['val'])}
      out=graph.call_function(operator.getitem,args=(new,0));out.meta={'val':old.meta['val']}
      rr=graph.call_function(operator.getitem,args=(new,1));rr.meta={'val':res.meta['val']}
     old.replace_all_uses_with(out);res.replace_all_uses_with(rr)
     for dead in (old,m['y'],res,norm):graph.erase_node(dead)
   graph.lint()
  after=graph.python_code('self').src
  if self.mode=='control' or (context.start,context.end)!=(64,64):assert before==after
  row=dict(index=self.index,mode=self.mode,compile_range=[context.start,context.end],bank_ids=[m['bank_id'] for m in matches],replaced=len(matches) if self.mode=='candidate' else 0,before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest(),all_other_ranges_unchanged=(before==after) if (context.start,context.end)!=(64,64) else None,source=str(Path(__file__).resolve()),source_sha256=sha(__file__))
  name=f'pass_{os.getpid()}_{self.index:03d}_{context.start}_{context.end}';self.index+=1;(self.directory/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');(self.directory/(name+'.py')).write_text(after)
