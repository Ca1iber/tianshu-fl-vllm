# SPDX-License-Identifier: Apache-2.0
"""Compose after qualified M64 pass; rewrite only the dynamic linear/SiLU pair."""
import hashlib,json,os
from pathlib import Path
import torch
from vllm.compilation.passes.inductor_pass import InductorPass,get_pass_context
from range_match import match

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class RangePass(InductorPass):
 def __init__(self,parent,mode,directory,selection):
  self.parent,self.mode,self.directory,self.selection=parent,mode,Path(directory),Path(selection);self.directory.mkdir(parents=True,exist_ok=False);self.index=0
 def uuid(self):return self.hash_dict(dict(parent=str(self.parent.uuid()),mode=self.mode,pass_source=sha(__file__),match_source=sha(Path(__file__).with_name('range_match.py')),selection=sha(self.selection),directory=str(self.directory)))
 def __call__(self,graph):
  self.parent(graph);context=get_pass_context().compile_range;bounds=[context.start,context.end];before=graph.python_code('self').src;matches=[]
  if bounds==[1,4096]:
   matches=[item for node in list(graph.nodes) if (item:=match(node)) is not None]
   if self.mode=='candidate':
    for item in matches:
     with graph.inserting_before(item['node']):new=graph.call_function(torch.ops.tianshu_m_range_prefill.fused_or_original.default,args=item['node'].args)
     new.meta=dict(item['out'].meta);item['out'].replace_all_uses_with(new)
     for node in item['remove']:graph.erase_node(node)
   graph.lint()
  after=graph.python_code('self').src
  if self.mode=='control' or bounds!=[1,4096]:assert before==after
  row=dict(mode=self.mode,index=self.index,compile_range=bounds,bank_ids=[item['bank_id'] for item in matches],replaced=len(matches) if self.mode=='candidate' else 0,before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest(),M64_parent_pass_retained=True,pass_source=sha(__file__))
  name=f'pass_{os.getpid()}_{self.index:03d}_{context.start}_{context.end}';self.index+=1
  (self.directory/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');(self.directory/(name+'.py')).write_text(after)
