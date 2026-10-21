# SPDX-License-Identifier: Apache-2.0
"""CPU INT64/F64 replay of 42-bank MLP actual GPU samples; explicit scope."""
import csv,hashlib,json,statistics
from pathlib import Path
import torch
assert not torch.cuda.is_initialized();torch.set_num_threads(8)
BASE=Path(__file__).resolve().parent;root=Path('/workspace/benchmark-results/20261021_m_range_MLP42_GPU_tianshu')
r=json.loads((root/'result.json').read_text());assert r['complete'] and r['phase']=='complete' and len(r['cases'])==24*42*3
assert r['all_42_GPU_bank_full_hashes_before_after'] and len(r['bank_signatures'])==42
assert {(c['M'],c['phase'],c['layer']) for c in r['cases']}=={(m,p,i) for m in r['rows'] for p in range(3) for i in range(42)}
for path,digest in r['sources'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((root/'source_snapshot'/Path(path).relative_to('/')).read_bytes()).hexdigest()==digest
signatures={row['prefix']:row for row in json.loads((BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text())}
for bank in r['bank_signatures']:
 assert all(bank[key]==signatures[bank['prefix']][key] for key in ('original_BF16_sha256','packed_sha256','scales_sha256'))
fixtures={};count=0;artifacts=0
for c in r['cases']:
 assert c['full_timed_activation_check']['passed'] and c['full_timed_MLP_check']['passed'] and c['full_INT32_and_projected_bitwise'] and c['full_A_quantization_exact']
 if c['artifact'] is None:continue
 artifacts+=1
 key=c['fixture']
 if key not in fixtures:
    f=torch.load(root/key,map_location='cpu',weights_only=False);a=f['A_rows'].float();scale=a.abs().amax(-1).clamp_min(1.e-8)/127.;q=(a/scale[:,None]).round().clamp(-127,127).to(torch.int8)
    assert torch.equal(q,f['q_rows']) and torch.equal(scale,f['scale_rows']);fixtures[key]=f
 f=fixtures[key];bank=f['banks'][c['layer']];dot=f['q_rows'].to(torch.int64)@bank['W'].to(torch.int64).T
 projected=(dot.float()*f['scale_rows'][:,None]*bank['S'][None,:]).to(torch.bfloat16)
 p=torch.load(root/c['artifact'],map_location='cpu',weights_only=False)
 assert torch.equal(p['integer'].to(torch.int64),dot) and torch.equal(p['projected'].view(torch.int16),projected.view(torch.int16))
 g,u=projected[:,:16].float(),projected[:,16:].float();act=g/(1+torch.exp(-g))*u
 for name in ('baseline_activation_selected','fused_activation_selected'):
    assert torch.isfinite(p[name]).all() and ((p[name].float()-act).abs()<=.001+.02*act.abs()).all();count+=p[name].numel()
 for prefix in ('baseline','fused'):
    expected=(p[prefix+'_activation_full_rows'].double()@bank['Down_W'].double().T).to(torch.bfloat16).float();actual=p[prefix+'_Down']
    assert torch.isfinite(actual).all() and ((actual.float()-expected).abs()<=.02+.01*expected.abs()).all(),(c['M'],c['phase'],c['layer'],prefix)
    count+=actual.numel()
 count+=dot.numel()+projected.numel()
assert artifacts==24*3*4
rows=[]
for t in r['timing_rounds']:
 assert t['weight_banks_per_graph']==42 and t['replays_per_trial']==3 and t['last_bank_output_after_timing']['passed']
 med={key:statistics.median(v) for key,v in t['times_ms_per_layer'].items()};assert med==t['medians_ms_per_layer']
 rows.append(dict(M=t['M'],**med,speedup_pct=(med['current_MLP']/med['fused_MLP']-1)*100,**r['configuration']))
assert len(rows)==24 and {x['M'] for x in rows}==set(r['rows'])
with (BASE/'MLP42_TIMINGS_32g.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
gate=dict(complete=True,CPU_only=True,component_only=True,model_executed=False,official_benchmark=False,adopted=False,cases=3024,actual_weight_banks=42,all_42_GPU_bank_full_hashes_before_after=True,CPU_actual_sample_artifacts=artifacts,CPU_selected_actual_GPU_elements=count,source_hashes=r['sources'],raw_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),scope='42 actual independent MLP weight banks sharing fixture X, not sequential transformer; full numerical GPU checks before timing; posttiming GPU output last bank only; offline CPU reload selected 4 actual layers, all layers online checked',timings=rows)
(BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
lines=['# 42实际权重bank、Inductor激活与完整MLP','', '24个当前prefill M × 42实际层 × 3输入相位 =3024场景通过；42份GPU W8/scale/Down BF16完整hash前后不变。INT32投影严格位相同，激活及最终MLP全GPU输出按预设阈值通过。CPU重载4层×24M×3相位的实际输出共'+str(count)+'选点元素，独立INT64投影与F64 Down均通过。','','对照为现用NN配置、提取的Inductor生成激活、原生BF16 Down（含M2048 special）。激活源函数/装饰器AST保持，组件独立重新编译，不称同服务binary。每个图轮换42真实bank，7组交替×3图回放，下面为每层时间；各层共用fixture X而非真实transformer，不是模型或正式成绩。','','|M|当前完整MLP ms/层|候选 ms/层|逆耗时变化|','|---:|---:|---:|---:|']
for c in rows:lines.append(f"|{c['M']}|{c['current_MLP']:.5f}|{c['fused_MLP']:.5f}|{c['speedup_pct']:.2f}%|")
lines+=['','是否进入服务需结合主要M的实际收益；不使用较慢FlagGems独立激活的35%组件值直接预测模型收益。尚未采用任何M范围融合，正式合格最好仍5618.01 total tok/s，采用必须真实4K与新Math。']
(BASE/'REPORT_32g.md').write_text('\n'.join(lines)+'\n');assert not torch.cuda.is_initialized();print(json.dumps({k:v for k,v in gate.items() if k not in ('source_hashes','timings')},indent=2));print((BASE/'REPORT_32g.md').read_text())
