# SPDX-License-Identifier: Apache-2.0
import hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (BASE/'selected_plan.json').exists();source=EXP/'tianshu_native_int8_attention_decode64_service_20261022/selected_plan.json';p=json.loads(source.read_text());rows=dict(QKV=[64,128,512,4096],O=[512,4096]);result=dict(p);result.update(rows_by_role=rows,block_by_M=dict(p['block_by_M']),expected_dequant_binaries=dict(p['expected_dequant_binaries']),QKV64_scope='FULL_uniform64_decode',QKV128_BF16=False,QKV128_native_retained=True,decode_scope_sha256=sha(BASE/'decode_scope.py'),new_full105_C8_C64_required=True,prior_failed915_quality_not_reused=True,prior928_quality_not_reused=True,only_extra_precision_vs903_r3='QKV64_FULL_uniform64_decode',selection_parent_source=str(source),selection_parent_sha256=sha(source));assert len(result['expected_dequant_binaries'])==6 and result['rows_by_role']==p['rows_by_role'];(BASE/'selected_plan.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
