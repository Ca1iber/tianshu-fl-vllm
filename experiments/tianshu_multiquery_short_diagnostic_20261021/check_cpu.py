# SPDX-License-Identifier: Apache-2.0
import ast,bisect,collections,hashlib,json
from pathlib import Path
import torch
from emulate import emulate
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'tianshu_multiquery_splitkv_20261021'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p in BASE.glob('*.py'):ast.parse(p.read_text())
configs=[dict(block_m=256,tile=128,warps=16,stages=1),dict(block_m=32,tile=32,warps=2,stages=1),dict(block_m=16,tile=16,warps=1,stages=1),dict(block_m=32,tile=16,warps=2,stages=1)]
cases=0
for ql,kl in ((2,2),(3,17),(4,33),(8,8),(8,65),(7,17)):
    q=torch.zeros(ql,16,128,dtype=torch.bfloat16);k=torch.zeros(1,kl,2,128,dtype=torch.bfloat16)
    v=torch.arange(kl,dtype=torch.float32)[None,:,None,None].expand(1,kl,2,128).to(torch.bfloat16).contiguous()
    for causal in (True,False):
      expected=torch.stack([v[0,:(kl-ql+pos+1 if causal else kl)].float().mean(0).repeat_interleave(8,0) for pos in range(ql)]).to(torch.bfloat16)
      for i,c in enumerate(configs):
        actual=emulate(q,k,v,[ql],[kl],c,i!=0,64,causal);assert torch.equal(actual,expected),(ql,kl,causal,c);cases+=1
mappings=0
for qlens in ([2]*64,[3]*64,[4]*64,[8]*64,[1,2,3,4,7,8,2,3]*8):
  for bq in (2,4):
    cu=[0]
    for q in qlens:cu.append(cu[-1]+q)
    prefix=[cu[i]//bq+i for i in range(len(qlens))];seen=collections.Counter()
    for block in range(cu[-1]//bq+len(qlens)):
      seq=bisect.bisect_right(prefix,block)-1;local=block-prefix[seq]
      for pos in range(local*bq,min((local+1)*bq,qlens[seq])):seen[cu[seq]+pos]+=1
    assert set(seen)==set(range(cu[-1])) and set(seen.values())=={1};mappings+=1
native=Path('/workspace/benchmark-results/20261020_k8v16_pair_load_GPU_tianshu/current_reader/generated_current.py')
assert 'acc += tl.dot(P.to(V.dtype), V)' in native.read_text()
original=Path('/workspace/benchmark-results/20261021_multiquery_splitkv_GPU_tianshu/result.json');raw=json.loads(original.read_text())
assert raw['phase']=='failed' and raw['error'].startswith("AssertionError(('q2_short_k2', 'control_2D', 0,")
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'PLAN_32g.md',OLD/'source.py',OLD/'launch.py',native,
    Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_attention_helpers.py'),BASE.parent/'tianshu_k8v16_pipeline_20261019/value_planes.py']
sources={str(p):sha(p) for p in paths}
result=dict(complete=True,CPU_only=True,GPU_not_used=True,emulated_exact_uniform_mean_cases=cases,query_mapping_cases=mappings,
    original_failed_result_sha256=sha(original),original_failure_preserved=True,original_ideal_FP32_threshold_unchanged=True,native_BF16_P_cast_verified=True,sources=sources)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
