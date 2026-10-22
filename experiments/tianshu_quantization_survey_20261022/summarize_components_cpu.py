# SPDX-License-Identifier: Apache-2.0
"""Combine only independently audited component results; no GPU imports."""
import csv,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
EXPERIMENTS=[('逐元素解包','tianshu_quant_weight_component_v2_20261022'),('成对读取解包','tianshu_quant_weight_pairload_20261022'),('W8 NN后缩放','tianshu_w8a16_postscale_20261022'),('W4 NN分组后缩放','tianshu_w4a16_group_postscale_20261022'),('W8 NT后缩放','tianshu_w8a16_nt_postscale_20261022'),('W4 NT分组后缩放','tianshu_w4a16_nt_group_postscale_20261022')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 output=[];evidence=[];lines=['# 天数权重量化组件汇总，2026-10-22','','原checkpoint只读，未转换架构、未微调。比较BF16 activation下的4/8bit权重存储及实际解包/GEMM成本；模型原文件前后SHA保持。全部候选目前未接模型、未跑新Math/正式benchmark；原正式发布5695.95 total tok/s保留。','','|独立组件版本|Down/M64 最快格式、ms|LM/M8 最快格式、ms|LM/M64 最快格式、ms|','|---|---|---|---|']
 for label,folder in EXPERIMENTS:
  b=ROOT/folder;g=json.loads((b/'INDEPENDENT_CPU_GATE_32g.json').read_text());state=json.loads((b/'GPU_GATE_32g.json').read_text());assert g['complete'] and state['complete'] and g['raw_result_sha256']==sha(b/'GPU_GATE_32g.json');assert g['key_timings_sha256']==sha(b/'KEY_TIMINGS_32g.csv');assert all(sha(p)==d for p,d in state['frozen_sources'].items())
  rows=list(csv.DictReader((b/'KEY_TIMINGS_32g.csv').open()));points=[]
  for role,m in [('down',64),('lm_head',8),('lm_head',64)]:
   candidates=[r for r in rows if (r['role'],int(r['M']))==(role,m)];r=min(candidates,key=lambda r:float(r['candidate_ms']));kind='W8A8' if role=='lm_head' and m==64 else 'BF16';control=float(r[f'{kind}_ms']);candidate=float(r['candidate_ms']);points.append(f"{r['format']} / {candidate:.6f}")
   output.append(dict(experiment=folder,label=label,role=role,M=m,format=r['format'],tile_id=int(r['tile_id']),candidate_ms=candidate,adopted_control=kind,control_ms=control,candidate_vs_adopted_duration=candidate/control,component_faster_than_adopted=(candidate<control)))
  lines.append('|'+label+'|'+'|'.join(points)+'|');evidence.append(dict(experiment=folder,CPU_gate_sha256=sha(b/'INDEPENDENT_CPU_GATE_32g.json'),GPU_gate_sha256=sha(b/'GPU_GATE_32g.json'),cases=len(state['cases']),phase_artifacts=g['phase_artifacts_verified'],decode_phases=g['independent_full_decode_phases'],selected_FP64_outputs=g['independent_selected_FP64_outputs']))
 assert not any(r['component_faster_than_adopted'] for r in output)
 lines+=['','每点为3个预定tile的7轮×20次热Graph中位数中最小值；完整格式/tile数据在各目录KEY_TIMINGS。当前已采用Down/LM8的BF16、LM64的W8A8，不能用较慢BF16替代已采用的LM64控制来宣称收益。activation为合成BF16，weight来自实际checkpoint第0层Down/LMhead；并非完整模型benchmark。','','全部列出的组件候选均未超过对应现用路径。结论局限于这些实现、shape和tile；不能推断天数所有INT4实现均不可加速。成对读取、NT布局确实减少了原型成本，但仍有显著差距。已安排IXKN实际kernel计数器定位，计数器采集结果另行报告。','','格式载荷（包括FP32 scale/uint8 zero）为BF16的26.6%–28.9%，与引擎显存净节省不同；保留BF16 fallback时属于新增副本。当前已采用INT8 GateUp/LM64及K8/V16 cache的质量/吞吐证据不借给新路径。','','6版均验证尾部/stridedA、3次变更同一图输入、完整format decode位级、全输出FP32/BF16 cast、完整GPU参考、独立CPU FP64抽查、所有A8量化和selected W8A8整数结果、guards/RNG/输入不变。原V1 signed-zero测试参考错误与离线V1序列化错误另存，不覆盖失败记录。','','GPU每日北京时间06:00–次日00:00，本日23:58截止。完整数组和profile仅容器，GitHub保留代码、报告和关键结果。']
 (BASE/'COMPONENT_SUMMARY_32g.md').write_text('\n'.join(lines)+'\n')
 with (BASE/'COMPONENT_SUMMARY_32g.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(output[0]),lineterminator='\n');writer.writeheader();writer.writerows(output)
 gate=dict(complete=True,CPU_only=True,experiments=evidence,component_candidates_adopted=0,model_quality_not_executed=True,official_benchmark_not_executed=True,summary_CSV_sha256=sha(BASE/'COMPONENT_SUMMARY_32g.csv'))
 (BASE/'COMPONENT_SUMMARY_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(dict(experiments=6,total_cases=sum(e['cases'] for e in evidence),total_phase_artifacts=sum(e['phase_artifacts'] for e in evidence),all_slower_than_adopted=True)))
if __name__=='__main__':main()
