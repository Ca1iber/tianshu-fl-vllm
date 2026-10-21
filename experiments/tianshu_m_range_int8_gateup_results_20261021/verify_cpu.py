# SPDX-License-Identifier: Apache-2.0
"""Recompute actual GPU integer/projection/activation samples from CPU fixtures."""
import csv,hashlib,json,statistics
from pathlib import Path
import torch
assert not torch.cuda.is_initialized();torch.set_num_threads(8)
BASE=Path(__file__).resolve().parent;root=Path('/workspace/benchmark-results/20261021_m_range_INT8_GateUp_GPU_tianshu')
r=json.loads((root/'result.json').read_text());assert r['complete'] and r['phase']=='complete' and len(r['cases'])==24*8
for name,digest in r['sources'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((root/'source_snapshot'/Path(name).relative_to('/')).read_bytes()).hexdigest()==digest
count=0;records=[];fixtures={}
for c in r['cases']:
    assert c['complete'] and len(c['phases'])==3 and c['timed_outputs_after_timing_bitwise']
    for ph in c['phases']:
        assert all(ph[key] for key in ('full_timed_BF16_bitwise','full_INT32_projection_bitwise','full_quantization_exact','CPU_selected_INT64_exact','CPU_activation_tolerance','guards_inputs_unchanged'))
        key=(c['M'],ph['phase'])
        if key not in fixtures:
            fixture=torch.load(root/f'fixture_M{key[0]}_phase{key[1]}.pt',map_location='cpu',weights_only=False)
            a=fixture['A'].float();scale=a.abs().amax(-1).clamp_min(1.e-8)/127.;q=(a/scale[:,None]).round().clamp(-127,127).to(torch.int8)
            assert torch.equal(q,fixture['q']) and torch.equal(scale,fixture['scale'])
            dot=q[fixture['rows']].to(torch.int64)@fixture['W'].to(torch.int64).T
            proj=(dot.float()*scale[fixture['rows'],None]*fixture['W_scale'][None,:]).to(torch.bfloat16)
            gate,up=proj[:,:16].float(),proj[:,16:].float();out=gate/(1+torch.exp(-gate))*up
            fixtures[key]=(dot,proj,out)
        dot,proj,out=fixtures[key];actual=torch.load(root/ph['artifact'],map_location='cpu',weights_only=False)
        assert torch.equal(actual['integer'].to(torch.int64),dot)
        assert torch.equal(actual['projected'].view(torch.int16),proj.view(torch.int16))
        assert torch.isfinite(actual['output']).all() and ((actual['output'].float()-out).abs()<=.001+.02*out.abs()).all()
        count+=dot.numel()+proj.numel()+out.numel()
    med={name:statistics.median(v) for name,v in c['times_ms'].items()}
    assert med==c['medians_ms']
    records.append(dict(M=c['M'],config_id=c['config_id'],current_model_INT8_enabled=c['current_model_INT8_enabled'],**c['configuration'],**med,speedup_vs_current_pct=(med['current_total']/med['candidate_total']-1)*100,speedup_vs_unfused_pct=(med['NN_unfused_total']/med['candidate_total']-1)*100))
selected=[max([c for c in records if c['M']==m],key=lambda c:c['speedup_vs_current_pct']) for m in r['rows']]
for name,data in [('ALL_CONFIGURATIONS_32g.csv',records),('BEST_BY_M_32g.csv',selected)]:
 with (BASE/name).open('w',newline='') as s:
    w=csv.DictWriter(s,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
gate=dict(complete=True,CPU_only=True,component_only=True,adopted=False,model_executed=False,official_benchmark=False,fixtures=len(fixtures),cases=len(records),CPU_selected_actual_GPU_elements=count,scope='GPU all INT32/projection/timed BF16 online bitwise; CPU reload only actual saved selected outputs, not all full outputs',source_hashes=r['sources'],raw_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),best_by_M=selected)
(BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
lines=['# GateUp M范围：布局、分组与融合','', '所有192配置×3相位正确性通过；CPU独立重算保存的实际GPU选点，完整GPU输出在线位相同。M64对照保留已采用融合，其他M对照分为当前已启用INT8与仅INT8子系统探索，后者不能当模型收益。','','|M|当前已用INT8|BM/BN/G|当前链路ms|候选ms|相对当前子系统|','|---:|---|---|---:|---:|---:|']
for c in selected:lines.append(f"|{c['M']}|{c['current_model_INT8_enabled']}|{c['bm']}/{c['bn']}/{c['group_m']}|{c['current_total']:.5f}|{c['candidate_total']:.5f}|{c['speedup_vs_current_pct']:.2f}%|")
lines+=['','均为单层固定权重热缓存组件。采用前仍需邻近M/42层实际权重与完整MLP/4K服务及新Math质量。实际weights/layout/sign/scale、动态A8量化和SiLU都没有简化；不采用已知错误single_dot。W NN bank当前已经存在，复用不需额外新bank。NT/NN同tile、量化/GEMM/激活独立时序见原result.stage_timings。']
(BASE/'REPORT_32g.md').write_text('\n'.join(lines)+'\n')
assert not torch.cuda.is_initialized();print(json.dumps({k:v for k,v in gate.items() if k not in ('source_hashes','best_by_M')},indent=2));print((BASE/'REPORT_32g.md').read_text())
