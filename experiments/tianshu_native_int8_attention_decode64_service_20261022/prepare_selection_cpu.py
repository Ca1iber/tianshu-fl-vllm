# SPDX-License-Identifier: Apache-2.0
import hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (BASE/'selected_plan.json').exists();parent=read(EXP/'tianshu_native_int8_attention_service_20261022/selected_plan.json');directory=EXP/'tianshu_native_int8_attention64_20261022';gate=read(directory/'INDEPENDENT_CPU_GATE_32g.json');assert gate['complete'] and gate['artifacts_checked']==168 and gate['independent_INT64_exact_BF16_and_FP64_control_cells']==43008;raw=Path(gate['raw_directory']);r=read(raw/'result.json');assert r['complete'] and sha(raw/'result.json')==gate['raw_result_sha256'] and r['banks']==parent['banks'] and r['BF16_control']=='actual_903source_service_explicit_vendor_mm' and [c['M'] for c in r['cases']]==[64]
 qkv=next(t for t in gate['timings'] if t['M']==64 and t['scope']=='QKV');assert qkv['speedup_pct']>1
 result=dict(parent);result.update(rows_by_role=dict(QKV=[64,128,512,4096],O=[512,4096]),block_by_M={**parent['block_by_M'],'64':512},expected_dequant_binaries={**parent['expected_dequant_binaries'],'QKV:64':r['cases'][0]['compiled']['QKV']['binary_sha256']},new_M64_component=dict(directory=str(directory),gate_sha256=sha(directory/'INDEPENDENT_CPU_GATE_32g.json'),raw_result_sha256=gate['raw_result_sha256'],QKV_speedup_pct=qkv['speedup_pct'],actual_control='903source_explicit_vendor_mm'),new_full105_C8_C64_required=True,prior903_quality_not_reused=True)
 assert len(result['expected_dequant_binaries'])==6 and result['quality_not_yet_established'];(BASE/'selected_plan.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in result.items() if k not in ('banks','holdouts','compiled_BF16_lowering_evidence')}))
if __name__=='__main__':main()
