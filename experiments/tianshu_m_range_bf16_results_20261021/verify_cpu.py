# SPDX-License-Identifier: Apache-2.0
"""Independent CPU replay of saved BF16 samples and full O8 diagnostics."""
import csv,hashlib,json,statistics
from pathlib import Path
import torch
assert not torch.cuda.is_initialized();torch.set_num_threads(8)
BASE=Path(__file__).resolve().parent
root=Path('/workspace/benchmark-results/20261021_m_range_BF16_v2_GPU_tianshu')
r=json.loads((root/'result.json').read_text());assert r['complete'] and r['phase']=='complete' and len(r['cases'])==96
for p,d in r['frozen_sources'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==d
    assert hashlib.sha256((root/'source_snapshot'/Path(p).relative_to('/')).read_bytes()).hexdigest()==d
rows=[];count=0
for c in r['cases']:
    p=torch.load(root/f"{c['role']}_M{c['M']}_phase{c['phase']}.pt",map_location='cpu',weights_only=False)
    y=(p['A'].double()@p['W'].double().T).to(torch.bfloat16).float()
    for stage in ('outputs_before_timing','outputs_after_timing'):
        for label,x in p[stage].items():
            passed=bool(torch.isfinite(x).all() and ((x.float()-y).abs()<=.02+.01*y.abs()).all())
            assert passed or (label=='native_library' and not c['native_comparison_eligible'])
            count+=x.numel()
    for stage in ('full_output_GPU_checks','timed_full_output_GPU_checks'):
        assert all(v['passed'] or (label=='native_library' and not c['native_comparison_eligible']) for label,v in c[stage].items())
    rows.append(dict(role=c['role'],M=c['M'],phase=c['phase'],native_eligible=c['native_comparison_eligible'],**c['median_ms']))
q=json.loads((BASE.parent/'tianshu_m_range_partial_20261021/PARTIAL_GATE_32g.json').read_text());assert q['complete']
original=json.loads(Path('/workspace/benchmark-results/20261021_m_range_BF16_GPU_tianshu/result.json').read_text())
qkv=[c for c in original['cases'] if c['role']=='qkv'];assert len(qkv)==48
for c in qkv:rows.append(dict(role='qkv',M=c['M'],phase=c['phase'],native_eligible=True,**c['median_ms']))
diagroot=Path('/workspace/benchmark-results/20261021_m_range_native_diag_v2_GPU_tianshu');diag=json.loads((diagroot/'result.json').read_text());assert diag['complete']
diag_elements=0
for phase in range(2):
    p=torch.load(diagroot/f'phase{phase}.pt',map_location='cpu',weights_only=False)
    y=(p['A_expected'].double()@p['W_expected'].double().T).to(torch.bfloat16).float()
    assert torch.equal(p['A_expected'],p['actual_A_after']) and torch.equal(p['W_expected'],p['actual_W_after'])
    for name,x in p['outputs'].items():
        assert torch.isfinite(x).all() and ((x.float()-y).abs()<=.02+.01*y.abs()).all(),(phase,name)
        diag_elements+=x.numel()
summary=[]
for role in ('qkv','o','down'):
 for m in r['rows']:
    cs=[c for c in rows if c['role']==role and c['M']==m];assert len(cs)==2
    med={label:statistics.mean(c[label] for c in cs) for label in ('native_library','matched_NT','packed_W','packed_AW','A_pack_only')}
    summary.append(dict(role=role,M=m,native_eligible=all(c['native_eligible'] for c in cs),**med,W_vs_matched_pct=(med['matched_NT']/med['packed_W']-1)*100,AW_vs_matched_pct=(med['matched_NT']/med['packed_AW']-1)*100,W_vs_native_pct=(med['native_library']/med['packed_W']-1)*100))
for name,data in [('ROUNDS_32g.csv',rows),('SUMMARY_32g.csv',summary)]:
 with (BASE/name).open('w',newline='') as s:
    writer=csv.DictWriter(s,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
gate=dict(complete=True,CPU_only=True,component_only=True,adopted=False,model_executed=False,BF16_v2_cases=96,QKV_reused_verified_cases=48,CPU_selected_elements=count,CPU_full_diagnostic_elements=diag_elements,native_invalid_cases=r['native_invalid_cases'],original_failure_retained=True,original_failure_cause='unresolved; isolated exact fixture does not reproduce; actual SPLIT_K=1',GPU_all_elements_scope='Online only in frozen GPU driver; CPU full reload only 12 diagnostic O8 outputs, main range CPU reload selected samples',sources=r['frozen_sources'],raw_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),diagnostic_result_sha256=hashlib.sha256((diagroot/'result.json').read_bytes()).hexdigest(),summary=summary)
(BASE/'BF16_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
lines=['# 分角色 BF16 M 范围初筛','', 'QKV首轮48相位复用已CPU复核部分档案；O/Down新96相位完整通过，候选与原生在线全输出原阈值检查。独立CPU重载O/Down选点共'+str(count)+'元素；O8完整诊断'+str(diag_elements)+'元素通过，真实SPLIT_K=1，原失败原因仍未确定，原failed保留。','','静态W额外内存按42层：QKV 420 MiB，O 336 MiB，Down 1008 MiB；AW还包含每步A转换。下面仅列各角色对同tile普通NT最好的5个W重排点，不推断整段范围、不直接当模型收益。','','|Role|M|W重排相对同tile NT|W重排相对库|AW相对同tile NT|','|---|---:|---:|---:|---:|']
for role in ('qkv','o','down'):
 for c in sorted([c for c in summary if c['role']==role],key=lambda x:x['W_vs_matched_pct'],reverse=True)[:5]:
    lines.append(f"|{role}|{c['M']}|{c['W_vs_matched_pct']:.2f}%|{c['W_vs_native_pct']:.2f}%|{c['AW_vs_matched_pct']:.2f}%|")
lines+=['','M576等mask尾块有明显性能悬崖；一些相对库的收益主要来自tile配置而非重排。后续需要变tile、邻近M加密、42真实权重轮换/冷热缓存，再决定按角色的范围白名单。当前INT8 GateUp已是NN布局，另测unfused grouped NN与row-major/grouped两dot融合，不拿BF16慢基准冒充当前模型。']
(BASE/'REPORT_32g.md').write_text('\n'.join(lines)+'\n')
assert not torch.cuda.is_initialized()
print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','summary')},indent=2))
print((BASE/'REPORT_32g.md').read_text())
