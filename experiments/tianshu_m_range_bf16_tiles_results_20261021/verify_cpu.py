# SPDX-License-Identifier: Apache-2.0
"""Independent actual BF16 source/fixture/sample verification and point-level rankings."""
import csv,hashlib,json,statistics
from pathlib import Path
import torch
from safetensors import safe_open
assert not torch.cuda.is_initialized();torch.set_num_threads(8)
BASE=Path(__file__).resolve().parent;root=Path('/workspace/benchmark-results/20261021_m_range_BF16_tiles_GPU_tianshu')
r=json.loads((root/'result.json').read_text());assert r['complete'] and r['phase']=='complete' and len(r['cases'])==504
assert {(c['role'],c['M'],c['config_id'],c['phase']) for c in r['cases']}=={(role,m,i,phase) for role in ('qkv','o','down') for m in r['rows'] for i in range(6) for phase in range(2)}
for name,digest in r['sources'].items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((root/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==digest
counts=0;records=[];weights={};inputs={}
with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as f:
 for role in ('qkv','o','down'):
    keys=(['model.layers.0.self_attn.'+s+'_proj.weight' for s in ('q','k','v')] if role=='qkv' else ['model.layers.0.self_attn.o_proj.weight'] if role=='o' else ['model.layers.0.mlp.down_proj.weight'])
    weights[role]=torch.cat([f.get_tensor(key) for key in keys],0).contiguous()
for c in r['cases']:
 role,m,phase=c['role'],c['M'],c['phase'];wc=weights[role];n,k=wc.shape;factor=1. if phase==0 else -.5
 payload=torch.load(root/c['artifact'],map_location='cpu',weights_only=False)
 if (role,m) not in inputs:
    ac=torch.randn(m,k,dtype=torch.bfloat16,generator=torch.Generator().manual_seed(202610212000+m+len(role)));ac[0].zero_();inputs[role,m]=ac[torch.tensor(sorted({0,1,m//2,m-1}))].clone()
 cols=torch.linspace(0,n-1,16).long()
 assert torch.equal(payload['A'],inputs[role,m]*factor) and torch.equal(payload['W'],wc[cols]*factor)
 y=(payload['A'].double()@payload['W'].double().T).to(torch.bfloat16).float()
 for stage in ('before','after'):
  for name,x in payload[stage].items():
    passed=bool(torch.isfinite(x).all() and ((x.float()-y).abs()<=.02+.01*y.abs()).all())
    assert passed or (name in ('native','current') and not c['reference_eligible'][name]),(role,m,c['config_id'],phase,stage,name)
    assert c[stage][name]['passed'] or (name in ('native','current') and not c['reference_eligible'][name])
    counts+=x.numel()
 med={name:statistics.median(t) for name,t in c['times_ms'].items()};assert med==c['medians_ms']
 records.append(dict(role=role,M=m,config_id=c['config_id'],phase=phase,**c['configuration'],native_eligible=c['reference_eligible']['native'],current_eligible=c['reference_eligible']['current'],**med))
summary=[]
for role in ('qkv','o','down'):
 for m in r['rows']:
  for i in range(6):
    cs=[c for c in records if c['role']==role and c['M']==m and c['config_id']==i];assert len(cs)==2
    med={name:statistics.mean(c[name] for c in cs) for name in ('native','current','matched_NT','packed_W')}
    summary.append(dict(role=role,M=m,config_id=i,configuration=r['configurations'][i],current_eligible=all(c['current_eligible'] for c in cs),**med,raw_vs_current_pct=(med['current']/med['matched_NT']-1)*100,packed_vs_current_pct=(med['current']/med['packed_W']-1)*100,W_vs_same_tile_pct=(med['matched_NT']/med['packed_W']-1)*100))
selected=[]
for role in ('qkv','o','down'):
 for m in r['rows']:
    options=[c for c in summary if c['role']==role and c['M']==m and c['current_eligible']]
    if not options:continue
    best_raw=max(options,key=lambda c:c['raw_vs_current_pct']);best_packed=max(options,key=lambda c:c['packed_vs_current_pct'])
    selected.append(dict(role=role,M=m,raw_config_id=best_raw['config_id'],raw_gain_pct=best_raw['raw_vs_current_pct'],packed_config_id=best_packed['config_id'],packed_gain_pct=best_packed['packed_vs_current_pct'],packing_incremental_pct=(best_raw['matched_NT']/best_packed['packed_W']-1)*100,raw_ms=best_raw['matched_NT'],packed_ms=best_packed['packed_W'],current_ms=best_raw['current']))
for name,data in [('ROUNDS_32g.csv',records),('BEST_BY_ROLE_M_32g.csv',selected)]:
 with (BASE/name).open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
gate=dict(complete=True,CPU_only=True,component_only=True,model_executed=False,adopted=False,cases=504,CPU_selected_actual_GPU_elements=counts,fixture_A_W_checked_against_original_checkpoint_and_seed=True,original_current_M2048_special_preserved=True,invalid_current_phase_cases=sum(not c['current_eligible'] for c in records),full_output_scope='GPU online all outputs; offline CPU reload selected real outputs only',sources=r['sources'],raw_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),configurations=r['configurations'],all_point_configurations=summary,best_by_role_M=selected)
(BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
lines=['# BF16 分角色六tile与W重排初筛','', '504相位完整，CPU重算'+str(counts)+'个实际GPU选点并验证A/W来自指定checkpoint和seed。当前M2048的O/Down特例保留；候选普通NT与W重排分别选择，重排增益与tile选择分开。仅单层固定真实权重组件，未采用，不是模型或服务。','','|Role|M|最佳普通tile收益|最佳重排收益|重排较最佳普通增量|','|---|---:|---:|---:|---:|']
for c in selected:lines.append(f"|{c['role']}|{c['M']}|{c['raw_gain_pct']:.2f}%|{c['packed_gain_pct']:.2f}%|{c['packing_incremental_pct']:.2f}%|")
lines+=['','候选选择需M邻点/边界、42真实权重轮换和实际4K确认；不能由14点插值整段范围。W重排会额外占QKV420/O336/Down1008MiB（42层），普通tile调整不增加bank。原生或当前参考有误的点不参与收益选择。']
(BASE/'REPORT_32g.md').write_text('\n'.join(lines)+'\n');assert not torch.cuda.is_initialized()
print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','configurations','all_point_configurations','best_by_role_M')},indent=2));print((BASE/'REPORT_32g.md').read_text())
