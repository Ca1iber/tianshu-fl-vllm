# SPDX-License-Identifier: Apache-2.0
import hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (BASE/'selected_plan.json').exists();p=json.loads((EXP/'tianshu_native_int8_attention_decode64_service_20261022/selected_plan.json').read_text());rows=dict(QKV=[64,512,4096],O=[512,4096]);result=dict(p);result.update(rows_by_role=rows,block_by_M={k:v for k,v in p['block_by_M'].items() if k!='128'},expected_dequant_binaries={k:v for k,v in p['expected_dequant_binaries'].items() if int(k.split(':')[1]) in rows[k.split(':')[0]]},QKV64_scope='FULL_uniform64_decode',QKV128_BF16=True,decode_scope_sha256=sha(BASE/'decode_scope.py'),new_full105_C8_C64_required=True,prior_failed915_quality_not_reused=True);assert len(result['expected_dequant_binaries'])==5;(BASE/'selected_plan.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
