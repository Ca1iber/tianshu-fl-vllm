# SPDX-License-Identifier: Apache-2.0
import hashlib,json,os
from pathlib import Path
import torch
from vllm.compilation.passes.inductor_pass import InductorPass,get_pass_context
import native_prefill_ops

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class NativePrefillPass(InductorPass):
    def __init__(self,parent,mode,directory):
        self.parent,self.mode,self.directory=parent,mode,Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False);self.index=0
    def uuid(self):
        return self.hash_dict(dict(parent=str(self.parent.uuid()),mode=self.mode,source=sha(__file__),ops=sha(Path(__file__).with_name('native_prefill_ops.py')),directory=str(self.directory)))
    def __call__(self,graph):
        self.parent(graph)
        bounds=[get_pass_context().compile_range.start,get_pass_context().compile_range.end]
        before=graph.python_code('self').src;bank_ids=[]
        if bounds==[1,4096]:
            for node in list(graph.nodes):
                if node.op=='call_function' and node.target==torch.ops.tianshu_m_range_prefill.fused_or_original.default:
                    assert len(node.args)==5 and isinstance(node.args[4],int)
                    bank_ids.append(node.args[4]);node.target=torch.ops.tianshu_native_prefill.fused_or_original.default
            graph.lint()
        after=graph.python_code('self').src
        if bounds!=[1,4096]:assert before==after
        row=dict(mode=self.mode,index=self.index,compile_range=bounds,bank_ids=bank_ids,replaced=len(bank_ids),before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest(),parent_applied_first=True,other_ranges_unchanged=(bounds!=[1,4096]),source=str(Path(__file__).resolve()),source_sha256=sha(__file__))
        name=f'pass_{os.getpid()}_{self.index:03d}_{bounds[0]}_{bounds[1]}';self.index+=1
        (self.directory/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');(self.directory/(name+'.py')).write_text(after)
