# SPDX-License-Identifier: Apache-2.0
"""Independent qualified wide-tile checks; original/current M64 binary bound."""
import csv,hashlib,json,statistics
from pathlib import Path
import torch
assert not torch.cuda.is_initialized();torch.set_num_threads(8)
BASE=Path(__file__).resolve().parent;root=Path('/workspace/benchmark-results/20261021_nn_silu_large_tiles_v2_GPU_tianshu')
r=json.loads((root/'result.json').read_text());assert r['complete'] and r['phase']=='complete' and len(r['cases'])==240
assert {(c['M'],c['config_id']) for c in r['cases']}=={(m,i) for m in r['rows'] for i in range(30)}
for name,digest in r['sources'].items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((root/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==digest
original=json.loads((BASE.parent/'tianshu_nn_silu_two_dot_20261021/GPU_GATE_32g.json').read_text());original_binary=next(c['actual_timed_compiled']['binary_sha256'] for c in original['best_by_shape'] if c['M']==64 and not c['strided'])
fixtures={};rows=[];qualified=0;count=0;failures=[]
for c in r['cases']:
 assert c['complete']
 if not c['qualified']:
    assert c.get('compile_failed') or any(not p['passed'] for p in c['phases'])
    failures.append(dict(M=c['M'],config_id=c['config_id'],configuration=c['configuration'],compile_error=c.get('compile_error'),failed_phases=[p for p in c['phases'] if not p['passed']]))
    continue
 qualified+=1;assert len(c['phases'])==3 and c['after_timing']['passed']
 if c['M']==64:assert c['actual_current_compiled']['binary_sha256']==original_binary
 for ph in c['phases']:
    assert all(ph[k] for k in ('passed','reference_Graph_eager_BF16_bitwise','full_INT32_exact','full_projected_BF16_exact','full_quantization_exact','CPU_INT64_exact'))
    if c['M']==64:assert ph['M64_timed_bitwise'] and ph['M64_current_check']['max_abs']==0
    key=(c['M'],ph['phase'])
    if key not in fixtures:
        f=torch.load(root/f'fixture_M{key[0]}_phase{key[1]}.pt',map_location='cpu',weights_only=False);v=f['A'].float();scale=v.abs().amax(-1).clamp_min(1.e-8)/127.;q=(v/scale[:,None]).round().clamp(-127,127).to(torch.int8)
        assert torch.equal(q,f['q']) and torch.equal(scale,f['scale']);dot=q[f['rows']].to(torch.int64)@f['W'].to(torch.int64).T;proj=(dot.float()*scale[f['rows'],None]*f['W_scale'][None,:]).to(torch.bfloat16);g,u=proj[:,:16].float(),proj[:,16:].float();a=g/(1+torch.exp(-g))*u;fixtures[key]=(dot,proj,a)
    dot,proj,a=fixtures[key];actual=torch.load(root/ph['artifact'],map_location='cpu',weights_only=False)
    assert torch.equal(actual['integer'].to(torch.int64),dot) and torch.equal(actual['projected'].view(torch.int16),proj.view(torch.int16))
    for name in ('output','reference_output'):
        assert torch.isfinite(actual[name]).all() and ((actual[name].float()-a).abs()<=.001+.02*a.abs()).all();count+=actual[name].numel()
    if c['M']==64:assert torch.equal(actual['output'].view(torch.int16),actual['current_output'].view(torch.int16))
    count+=dot.numel()+proj.numel()
 med={name:statistics.median(v) for name,v in c['times_ms'].items()};assert med==c['medians_ms']
 rows.append(dict(M=c['M'],config_id=c['config_id'],**c['configuration'],**med,speedup_vs_current_pct=(med['current']/med['candidate']-1)*100,registers=c['compiled']['registers'],spills=c['compiled']['spills'],binary_sha256=c['compiled']['binary_sha256']))
best=[]
for m in r['rows']:
 options=[row for row in rows if row['M']==m]
 if options:best.append(max(options,key=lambda c:c['speedup_vs_current_pct']))
for name,data in [('QUALIFIED_TIMINGS_32g.csv',rows),('BEST_BY_M_32g.csv',best)]:
 if data:
  with (BASE/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
gate=dict(complete=True,CPU_only=True,component_only=True,model_executed=False,adopted=False,attempted=240,qualified=qualified,excluded=len(failures),CPU_selected_actual_GPU_elements=count,M64_original_current_binary_sha256=original_binary,M64_all_timed_outputs_bitwise_original=True,reference_dynamic_Graph_eager_all_phases_bitwise=True,source_hashes=r['sources'],raw_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),best_by_M=best,failures=failures,scope='Single actual layer hot-cache component, Graph dynamic/eager reference checked; GPU full numerical online, offline CPU selected outputs only; M64 baseline is exact deployed binary, other M current NN+generated Inductor activation')
(BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
lines=['# 宽tile融合 v2：实时stream、当前M64二进制对照','', f'240配置全部执行，{qualified}完整合格、{len(failures)}保留错误且排除。CPU独立重算{count}实际GPU选点，M64三个动态输入相位全输出位相同已采用版，current binary严格匹配。其他M使用实时stream的Inductor生成激活，reference Graph/eager每相位全位一致。单层热缓存组件，不是模型或正式成绩。','','|M|BM/BN/BK/W/G|当前ms|候选ms|逆耗时变化|','|---:|---|---:|---:|---:|']
for c in best:lines.append(f"|{c['M']}|{c['bm']}/{c['bn']}/{c['bk']}/{c['warps']}/{c['group_m']}|{c['current']:.5f}|{c['candidate']:.5f}|{c['speedup_vs_current_pct']:.2f}%|")
lines+=['','采用前需42层实际bank冷缓存/完整4K和新Math；M64与已采用版全位一致不替代正式质量评测。相同binary或单Mtile的group1/8不以微小计时噪声声称不同优化，先前fixed-stream轮已STOPPED，不能混用其数值错误或计时。']
(BASE/'REPORT_32g.md').write_text('\n'.join(lines)+'\n');assert not torch.cuda.is_initialized();print(json.dumps({k:v for k,v in gate.items() if k not in ('source_hashes','best_by_M','failures')},indent=2));print((BASE/'REPORT_32g.md').read_text())
