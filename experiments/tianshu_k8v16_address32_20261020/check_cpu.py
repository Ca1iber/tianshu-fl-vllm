# SPDX-License-Identifier: Apache-2.0
"""Source and signed-offset boundary evidence only; imports no torch or GPU backend."""
import argparse,hashlib,json,random
from pathlib import Path
from types import SimpleNamespace
import source
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.output.resolve();assert root.is_relative_to('/workspace/benchmark-results');root.mkdir(parents=True,exist_ok=False)
proofs={}
for mode in ('kv32','all32'):
    text,proof=source.generate(a.parent,mode);(root/('generated_'+mode+'.py')).write_text(text);proofs[mode]=proof
checks=[]
for label,shape,strides,expected in [
    ('actual_K',(43166,16,2,128),(12288,256,128,1),True),
    ('actual_V',(43166,16,2,128),(6144,256,128,1),True),
    ('Q_padded',(256,16,128),(2560,128,1),True),
    ('huge_shape',(2**31,),(1,),False),
    ('masked_margin_crosses_signed_limit',(2**31-2047,),(1,),False),
    ('last_safe_1d',(2**31-2048,),(1,),True),
    ('wide_stride',(1,),(2**31,),False),
    ('negative_stride',(64,),(-1,),False)]:
    tensor=SimpleNamespace(shape=shape,stride=lambda strides=strides:strides)
    actual=source.safe_tensor(tensor);assert actual==expected,(label,actual)
    checks.append(dict(case=label,shape=shape,strides=strides,safe=actual))
rng=random.Random(20261020);samples=0
for _ in range(2000):
    shape=(rng.randint(1,100000),16,2,128);stride=(rng.choice((12288,24576,49152)),256,128,1)
    bound=source.offset_bound(shape,stride)
    accepted=bound<2**31
    for _ in range(16):
        index=tuple(rng.randrange(n) for n in shape);offset=sum(i*s for i,s in zip(index,stride))
        if accepted:assert 0<=offset<2**31
        assert offset<=bound;samples+=1
base=Path(__file__).resolve().parent
sources=[Path(__file__),Path(source.__file__),a.parent]
result=dict(complete=True,CPU_only=True,GPU_executed=False,adopted=False,provenance=proofs,boundary_cases=checks,
            independent_integer_offset_samples=samples,no_model_or_GPU_performance_claim=True,
            sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(root/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
for path in sources:
    dst=root/'source_snapshot'/path.resolve().relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(path.read_bytes())
print(json.dumps(dict(complete=True,CPU_only=True,provenance=proofs,boundaries=len(checks),offset_samples=samples)))
