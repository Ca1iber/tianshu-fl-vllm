# SPDX-License-Identifier: Apache-2.0
"""CPU recurrence/reference proof, source reversal and exact current baseline binding."""
import argparse,hashlib,json,math,random
from pathlib import Path
import source
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.output.resolve();assert root.is_relative_to('/workspace/benchmark-results');root.mkdir(parents=True,exist_ok=False)
native=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py');proofs={}
for mode in source.MODES:
    text,proof=source.generate(native.read_bytes(),mode);(root/f'generated_{mode}.py').write_text(text);proofs[mode]=proof
rng=random.Random(20261021);checks=[]
for n in (17,129,1025,4097):
    normal=[rng.gauss(0,1) for _ in range(n)]
    corpora=dict(normal=normal,wide=[x*4 for x in normal],shift_up=[x+1000 for x in normal],shift_down=[x-1000 for x in normal],
                 peaked=[20. if i==n//2 else x for i,x in enumerate(normal)],ties=[round(x) for x in normal],
                 flat=[0.]*n,masked=[x if i%3 else -math.inf for i,x in enumerate(normal)],
                 one_finite=[1. if i==0 else -math.inf for i in range(n)],all_inf=[-math.inf]*n)
    for name,values in corpora.items():
        maximum=max(values);reference=math.fsum(math.exp(v-maximum) for v in values) if math.isfinite(maximum) else 0.
        for tile in (7,128,32768):
            online_max=-math.inf;online_sum=0.
            for start in range(0,n,tile):
                block=values[start:start+tile];previous=online_max;online_max=max(previous,max(block))
                scale=math.exp(previous-online_max) if math.isfinite(previous) else 0.
                local=math.fsum(math.exp(v-online_max) for v in block if math.isfinite(v))
                online_sum=online_sum*scale+local
            assert online_max==maximum
            relative=abs(online_sum-reference)/max(reference,1.)
            assert relative<=1e-12,(n,name,tile,relative)
            checks.append(dict(n=n,distribution=name,tile=tile,relative_error=relative,maximum_correct=True))
assert len(checks)==120
paths=[Path(__file__),*source.dependency_sources()]
result=dict(complete=True,CPU_only=True,GPU_executed=False,adopted=False,independent_online_sum_checks=len(checks),
            scope='FP64 recurrence and source proof only; no claim of FP32 GPU semantic equivalence, speed or model quality',
            sources={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},provenance=proofs,cases=checks)
(root/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(complete=True,checks=len(checks),provenance=proofs)))
