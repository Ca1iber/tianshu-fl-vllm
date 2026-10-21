# SPDX-License-Identifier: Apache-2.0
"""Read-only native post-grad graph observation; no graph transformations."""
import hashlib,json,os
from collections import Counter
from pathlib import Path
import torch
from vllm.compilation.passes.inductor_pass import InductorPass,get_pass_context

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class Observer(InductorPass):
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False);self.index=0
    def uuid(self):return self.hash_dict(dict(source=digest(__file__),kind='read_only',directory=str(self.directory)))
    def __call__(self,graph):
        before=graph.python_code('self').src
        current=get_pass_context().compile_range
        def enc(value):
            if isinstance(value,torch.fx.Node):return {'node':value.name}
            if isinstance(value,(list,tuple)):return [enc(v) for v in value]
            if isinstance(value,dict):return {str(k):enc(v) for k,v in value.items()}
            if value is None or isinstance(value,(int,float,bool,str)):return value
            return str(value)
        def row(node):
            value=node.meta.get('val');meta=None
            if isinstance(value,torch.Tensor):meta=dict(shape=[str(v) for v in value.shape],stride=[str(v) for v in value.stride()],dtype=str(value.dtype),device=str(value.device))
            return dict(name=node.name,op=node.op,target=str(node.target),args=enc(node.args),kwargs=enc(node.kwargs),users=[u.name for u in node.users],meta=meta)
        selected=[]
        for node in graph.nodes:
            if 'tianshu_prefill_int8.linear' not in str(node.target):continue
            seen={node};front={node}
            for _ in range(7):
                front={u for item in front for u in item.users}-seen;seen.update(front)
            selected.append(dict(linear=row(node),descendants=[row(n) for n in graph.nodes if n in seen and n is not node]))
        assert graph.python_code('self').src==before
        result=dict(read_only=True,no_graph_mutation=True,compile_range=[current.start,current.end],index=self.index,pid=os.getpid(),node_count=len(list(graph.nodes)),op_counts=dict(Counter(str(n.target) for n in graph.nodes if n.op=='call_function')),linear_neighborhoods=selected,source_sha256=hashlib.sha256(before.encode()).hexdigest(),observer_source_sha256=digest(__file__))
        name=f'graph_{os.getpid()}_{self.index:03d}_{current.start}_{current.end}';self.index+=1
        (self.directory/(name+'.py')).write_text(before)
        (self.directory/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
