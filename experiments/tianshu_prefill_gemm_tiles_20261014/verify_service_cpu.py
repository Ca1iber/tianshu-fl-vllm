# SPDX-License-Identifier: Apache-2.0
"""Execute the actual guarded M2048 model launch function with CPU fake tensors."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    base=Path(__file__).resolve().parent
    source=base/'hooks/sitecustomize.py'
    tree=ast.parse(source.read_text())
    launch=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='launch')
    calls=[]
    class Tensor:
        def __init__(self,dtype='bf16',contiguous=True,stride=(1,2048),device='gpu0'):
            self.dtype=dtype;self.contiguous=contiguous;self.strides=stride;self.device=device
        def is_contiguous(self):return self.contiguous
        def stride(self):return self.strides
        @property
        def T(self):return self
    def original(*args):calls.append(('original',None));return 'original'
    def candidate(*args):calls.append(('candidate',args[-1]));return args[2][1]
    env=dict(original=original,kernel=SimpleNamespace(mm=candidate),plans={(2048,2048):1,(12288,2048):8,(2048,6144):1},
             torch=SimpleNamespace(bfloat16='bf16'),recorded=set(),os=SimpleNamespace(environ={}))
    exec(compile(ast.Module(body=[launch],type_ignores=[]),str(source),'exec'),env)
    cases=0;routed=0
    for n,k in ((2048,2048),(12288,2048),(2048,6144),(2560,2048),(130560,2048)):
        for m in (64,1024,2048):
            for dtype in ('bf16','f32'):
                for contiguous in (True,False):
                    for stride in ((1,k),(k,1)):
                        a=Tensor(dtype,contiguous);b=Tensor(dtype,stride=stride);c=Tensor(dtype)
                        actual=env['launch'](a,b,c,m,n,k)
                        selected=m==2048 and (n,k) in env['plans'] and dtype=='bf16' and contiguous and stride==(1,k)
                        assert (actual is c)==selected
                        if selected:
                            assert calls[-1][1]==dict(bm=256,bn=256,bk=32,splits=1,warps=16,stages=2,group_m=env['plans'][n,k]);routed+=1
                        else:assert calls[-1][0]=='original'
                        cases+=1
    for change in ('out_dtype','out_contiguous','out_device','weight_device'):
        a,b,c=Tensor(),Tensor(),Tensor()
        if change=='out_dtype':c.dtype='f32'
        if change=='out_contiguous':c.contiguous=False
        if change=='out_device':c.device='gpu1'
        if change=='weight_device':b.device='gpu1'
        assert env['launch'](a,b,c,2048,2048,2048)=='original';cases+=1
    data=dict(complete=True,cpu_only=True,actual_model_launch_AST_exercised=True,
              guarded_cases=cases,routed_cases=routed,only_tested_M2048_projections_routed=True,
              dtype_layout_output_device_fallbacks_verified=True,gpu_model_service_not_run=True,
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,base/'experimental_mm.py',Path(__file__))})
    out=base/'results/20261014/service_prepared_32g';out.mkdir(parents=True,exist_ok=True)
    (out/'CPU_GUARD_AUDIT_32g.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data))


if __name__=='__main__':main()
