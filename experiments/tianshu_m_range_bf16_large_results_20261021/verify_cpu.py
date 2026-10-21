# SPDX-License-Identifier: Apache-2.0
"""Independent full schedule + real checkpoint fixtures + saved GPU samples for BF16 large-M sweep."""
import base64,csv,hashlib,importlib.util,json,math,os,statistics
from pathlib import Path
os.environ['VLLM_PLUGINS']=''
import torch
from safetensors import safe_open
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;PROBE=EXP/'tianshu_m_range_bf16_large_20261021'
ROOT=Path('/workspace/benchmark-results/20261021_m_range_BF16_large_GPU_tianshu')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
assert not torch.cuda.is_initialized();torch.set_num_threads(8)
spec=importlib.util.spec_from_file_location('_BF16_large_declared_schedule',PROBE/'probe_gpu.py');schedule=importlib.util.module_from_spec(spec);spec.loader.exec_module(schedule)
gate=read(PROBE/'CPU_GATE_32g.json');state=read(ROOT/'result.json')
assert state['complete'] and state['phase']=='complete' and state['sources']==gate['sources']
for p,h in gate['sources'].items():assert sha(p)==h==sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))
expected={(role,m,ci,phase) for role in ('qkv','o','down') for m in schedule.ROWS for ci in range(len(schedule.CONFIGS)) for phase in range(2)}
actual=[(r['role'],r['M'],r['config_id'],r['phase']) for r in state['cases']];assert len(actual)==len(set(actual))==1008 and set(actual)==expected
assert state['rows']==schedule.ROWS and state['configurations']==schedule.CONFIGS
samples=0;rows=[];lookup={};invalid=[];fixture_cache={}
with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as ck:
 weights={role:torch.cat([ck.get_tensor(k) for k in (['model.layers.0.self_attn.'+s+'_proj.weight' for s in ('q','k','v')] if role=='qkv' else ['model.layers.0.self_attn.o_proj.weight'] if role=='o' else ['model.layers.0.mlp.down_proj.weight'])],0).contiguous() for role in ('qkv','o','down')}
 for case in state['cases']:
  role,m,ci,phase=case['role'],case['M'],case['config_id'],case['phase'];cfg=schedule.CONFIGS[ci];assert case['configuration']==cfg
  factor=1. if phase==0 else -.5;w=weights[role];n,k=w.shape;selected_rows=torch.tensor(sorted({0,1,m//2,m-1}));selected_cols=torch.linspace(0,n-1,16).long()
  key=(role,m)
  if key not in fixture_cache:
   ac=torch.randn(m,k,dtype=torch.bfloat16,generator=torch.Generator().manual_seed(202610212000+m+len(role)));ac[0].zero_();fixture_cache[key]=(ac[selected_rows].clone(),w[selected_cols].clone());del ac
  expected_a=fixture_cache[key][0]*factor;expected_w=fixture_cache[key][1]*factor
  fixture=torch.load(ROOT/case['artifact'],map_location='cpu',weights_only=True)
  assert torch.equal(fixture['A'].view(torch.int16),expected_a.view(torch.int16)) and torch.equal(fixture['W'].view(torch.int16),expected_w.view(torch.int16))
  expected_c=(expected_a.double()@expected_w.double().T).to(torch.bfloat16);assert torch.equal(fixture['CPU_reference'].view(torch.int16),expected_c.view(torch.int16))
  for label in ('native','current','matched_NT','packed_W'):
   checks=[]
   for period in ('before','after'):
    x=fixture[period][label];assert x.shape==expected_c.shape and x.dtype==torch.bfloat16
    valid=bool(torch.isfinite(x).all() and ((x.float()-expected_c.float()).abs()<=.02+.01*expected_c.float().abs()).all());checks.append(valid);samples+=x.numel()
   if label in ('matched_NT','packed_W'):assert all(checks) and case['before'][label]['passed'] and case['after'][label]['passed'] and case['selected_checks'][label]['passed']
   else:
    eligibility=case['before'][label]['passed'] and case['after'][label]['passed'] and case['selected_checks'][label]['passed'];assert eligibility==case['reference_eligible'][label]
    if eligibility:assert all(checks)
    else:invalid.append(dict(role=role,M=m,config_id=ci,phase=phase,label=label))
   values=case['times_ms'][label];assert len(values)==7 and all(math.isfinite(v) and v>0 for v in values)
   assert math.isclose(statistics.median(values),case['medians_ms'][label],rel_tol=1.e-12)
  for label,record in case['compiled'].items():
   key=(role,m,ci,label);assert key not in lookup or lookup[key]==record;lookup[key]=record
   folder=base64.b32encode(bytes.fromhex(record['hash'])).decode().rstrip('=')
   candidates=[cache/folder/'matched_nt_kernel.cubin' for cache in (Path('/root/.triton/cache'),Path('/tmp/torchinductor_root/triton/0'))];assert any(p.exists() and sha(p)==record['binary_sha256'] for p in candidates)
  assert case['extra_W_bytes']==w.numel()*w.element_size()
  rows.append(dict(role=role,M=m,config_id=ci,phase=phase,bm=cfg['bm'],bn=cfg['bn'],bk=cfg['bk'],warps=cfg['warps'],group_m=cfg['group_m'],current_eligible=case['reference_eligible']['current'],native_eligible=case['reference_eligible']['native'],**{label+'_ms':case['medians_ms'][label] for label in ('native','current','matched_NT','packed_W')},extra_W_bytes=case['extra_W_bytes']))
best=[]
for role in ('qkv','o','down'):
 for m in schedule.ROWS:
  configs=[]
  for ci,cfg in enumerate(schedule.CONFIGS):
   selected=[r for r in rows if r['role']==role and r['M']==m and r['config_id']==ci];assert len(selected)==2
   if not all(r['current_eligible'] for r in selected):continue
   values={label:statistics.mean(r[label+'_ms'] for r in selected) for label in ('native','current','matched_NT','packed_W')}
   for impl in ('matched_NT','packed_W'):
    configs.append(dict(role=role,M=m,implementation=impl,configuration=cfg,current_ms=values['current'],candidate_ms=values[impl],improvement_percent=100*(values['current']/values[impl]-1),pack_increment_same_tile_percent=100*(values['matched_NT']/values['packed_W']-1),extra_W_bytes=(selected[0]['extra_W_bytes'] if impl=='packed_W' else 0)))
  if configs:best.append(max(configs,key=lambda x:x['improvement_percent']))
assert not torch.cuda.is_initialized() and len(fixture_cache)==36
result=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,component_only=True,adopted=False,official_benchmark=False,full_1008_schedule_and_unique_cases_verified=True,actual_GPU_samples_checked=samples,independent_seeded_fixture_generations=36,all_1008_fixture_comparisons_still_executed=True,real_checkpoint_and_seeded_A_fixtures_exact=True,all_compiled_candidate_binary_hashes_checked_against_real_cache=True,all_candidate_numerical_checks_passed=True,invalid_reference_cases=invalid,rows=best,raw_root=str(ROOT),raw_result_sha256=sha(ROOT/'result.json'),source_hashes=gate['sources'],verifier_source_sha256=sha(__file__),baseline_scope='Current actual model BF16 path including M2048 O/Down special; no cross-tile packing attribution')
assert not (BASE/'GPU_GATE_32g.json').exists();(BASE/'GPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
with (BASE/'ALL_TIMINGS_32g.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# BF16大M按角色分离tile、调度与W重排','', '单层组件；1008相位和实际保存样本独立CPU重算，输入fixture与真实checkpoint逐位一致；所有候选binary与实际缓存核对。当前基线保留M2048 O/Down真实特殊tile。负收益不采用，实际模型收益必须再验证42bank/4K及质量。','', '|role|M|实现|BM/BN/BK/W/G|当前ms|候选ms|相对当前|同tile重排增益|','|---|---:|---|---|---:|---:|---:|---:|']
for r in best:
 c=r['configuration'];lines.append(f"|{r['role']}|{r['M']}|{r['implementation']}|{c['bm']}/{c['bn']}/{c['bk']}/{c['warps']}/{c['group_m']}|{r['current_ms']:.5f}|{r['candidate_ms']:.5f}|{r['improvement_percent']:+.2f}%|{r['pack_increment_same_tile_percent']:+.2f}%|")
lines+=['',f'无效参考相位 {len(invalid)} 个，明确排除对应参考；候选全部通过。W重排额外权重只计组件单bank字节，采用前必须计入42bank和KV cache容量。']
(BASE/'REPORT_32g.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_hashes','invalid_reference_cases')}))
