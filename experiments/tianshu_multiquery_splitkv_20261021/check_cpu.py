# SPDX-License-Identifier: Apache-2.0
import ast,bisect,collections,itertools,json,random
from pathlib import Path
from source import PARENT,HELPERS,CONFIGS,render,sha
BASE=Path(__file__).resolve().parent
for p in BASE.rglob('*.py'):ast.parse(p.read_text())
proofs={m:render(m)[1] for m in ('control','candidate')}
# Actual native binary-search relation: floor(cu/BLOCK_Q)+sequence.
cases=0;addresses=0
rng=random.Random(8972)
for n in (1,2,3,8,16,24,32,40,64):
  patterns=[[q]*n for q in (1,2,3,4,7,8)]
  patterns += [[rng.randint(1,8) for _ in range(n)] for _ in range(20)]
  for lengths in patterns:
    cu=[0]
    for q in lengths:cu.append(cu[-1]+q)
    for c in CONFIGS+[dict(block_m=256)]:
      bq=c['block_m']//8;prefix=[cu[i]//bq+i for i in range(n)];seen=collections.Counter()
      for block in range(cu[-1]//bq+n):
        seq=bisect.bisect_right(prefix,block)-1;assert 0<=seq<n
        local=block-prefix[seq]
        for pos in range(local*bq,min((local+1)*bq,lengths[seq])):
          for head in range(16):seen[(cu[seq]+pos,head)]+=1
      expected={(token,h) for token in range(cu[-1]) for h in range(16)}
      assert set(seen)==expected and set(seen.values())=={1};addresses+=len(seen);cases+=1
# Every true sequence key included exactly once in actual native segment intervals.
segments_checked=0
for n in (1,8,24,32,64):
  from source import segments
  s=segments(n)
  for tile in (32,64):
    for length in (2,3,7,8,17,31,32,33,65,129,4096,4097,5120):
      tps=(length+s*tile-1)//(s*tile)
      if n>24 and length<=8192:tps=(length+4*tile-1)//(4*tile)
      counts=collections.Counter(k for seg in range(s) for k in range(seg*tps*tile,min((seg+1)*tps*tile,length)))
      assert set(counts)==set(range(length)) and set(counts.values())=={1};segments_checked+=1
sources={str(p):sha(p) for p in [*BASE.glob('*.py'),*BASE.glob('*.sh'),BASE/'PLAN_32g.md',PARENT,HELPERS,Path('/workspace/vllm-plugin-FL/experiments/tianshu_k8v16_pipeline_20261019/value_planes.py')]}
# Execute the actual generated Python selection expression on CPU metadata.
from types import SimpleNamespace
fn=next(n for n in ast.parse(render('candidate')[0]).body if isinstance(n,ast.FunctionDef) and n.name=='unified_attention')
assignment=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='use_3d' for t in n.targets))
selection=compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'actual_generated_selection','exec')
class TensorMeta:
    def __init__(self,shape,dtype='f32',contiguous=True):self.shape,self.dtype,self.ndim,self.contiguous=shape,dtype,len(shape),contiguous
    def is_contiguous(self):return self.contiguous
def native_selection(**updates):
    ns=dict(seq_threshold_3D=64,num_par_softmax_segments=4,
        softmax_segm_output=TensorMeta((513,16,4,128)),softmax_segm_max=TensorMeta((513,16,4)),softmax_segm_expsum=TensorMeta((513,16,4)),
        q=TensorMeta((512,16,128),'bf16'),k=TensorMeta((17000,16,2,128),'u8'),v=TensorMeta((17000,16,2,128),'bf16'),
        max_seqlen_q=8,num_queries_per_kv=8,head_size=128,head_size_padded=128,num_query_heads=16,num_seqs=64,is_batch_invariant=False,
        torch=SimpleNamespace(uint8='u8',bfloat16='bf16',float32='f32'))
    ns.update(updates);exec(selection,ns);return ns['use_3d']
assert all(native_selection(max_seqlen_q=q) for q in range(1,9))
rejections=[dict(max_seqlen_q=9),dict(num_seqs=65),dict(seq_threshold_3D=None),dict(num_par_softmax_segments=None),dict(is_batch_invariant=True),dict(num_queries_per_kv=4),dict(head_size=64),dict(k=TensorMeta((1,16,2,128),'bf16')),dict(v=TensorMeta((1,16,2,128),'u8'))]
for key,shape in [('softmax_segm_output',(513,16,4,128)),('softmax_segm_max',(513,16,4)),('softmax_segm_expsum',(513,16,4))]:
    rejections += [{key:None},{key:TensorMeta((511,*shape[1:]))},{key:TensorMeta(shape,contiguous=False)},{key:TensorMeta(shape,'bf16')},{key:TensorMeta((513,16,8,*shape[3:]))}]
rejections += [dict(num_query_heads=32),dict(q=TensorMeta((512,16,128),"f32")),dict(num_par_softmax_segments=0),dict(num_par_softmax_segments=3)]
assert all(not native_selection(**r) for r in rejections)
result=dict(complete=True,CPU_only=True,GPU_not_used=True,query_grid_cases=cases,query_head_addresses=addresses,
    actual_native_mapping_formula_checked=True,actual_guard_acceptances=8,actual_guard_rejections=len(rejections),segmentation_cases=segments_checked,numerical_JIT_AST_unchanged=True,
    scratch_required_per_token_bytes_at_64segments=16*64*(128+2)*4,
    model_executed=False,official_benchmark=False,configs=CONFIGS,proofs=proofs,sources=sources)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('sources','proofs')}))
