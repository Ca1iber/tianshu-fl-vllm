# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261021_multiquery_short_diagnostic_GPU_tianshu')
SOURCE=BASE.parent/'tianshu_multiquery_short_diagnostic_20261021/emulate.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('_offline_short_cpu_emulation',SOURCE);emu=importlib.util.module_from_spec(spec);spec.loader.exec_module(emu)
import torch
torch.set_num_threads(8)
r=json.loads((ROOT/'result.json').read_text());assert r['complete'] and r['phase']=='complete' and len(r['cases'])==6
for p,h in r['frozen_sources'].items():assert sha(p)==h and sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))==h
checks=0;ideal_failed_modes=[];exact_control_equivalence=[]
for i,row in enumerate(r['cases']):
    p=ROOT/f'short_{i}.pt';assert sha(p)==row['archive_sha256'];a=torch.load(p,map_location='cpu',weights_only=True)
    ql,kl=row['query_length'],row['key_length'];assert len(row['modes'])==4
    original=a['original_K'];bits=original.view(torch.int16).int();mag=bits&32767;z=((mag+7+((mag>>4)&1))>>4)-928
    enc=torch.where(z>=8,z.clamp(max=127),0)|((bits&32768)>>8);m=enc&127
    decoded=(torch.where(m>=8,(m<<4)+14848,0)|((enc&128)<<8)).to(torch.int16).view(torch.bfloat16);assert torch.equal(decoded,a['decoded_K'])
    for mode in row['modes']:
        actual=a['outputs'][mode['label']];expected=emu.emulate(a['q'],decoded,a['V'],[ql]*8,[kl]*8,mode['config'],mode['label']!='control',16,row['causal'])
        assert not bool(((actual.float()-expected.float()).abs()>(.002+.01*expected.float().abs())).any())
        ideal=a['ideal'];bad=(actual.float()-ideal).abs()>(.002+.01*ideal.abs())
        assert int(bad.sum())==mode['ideal_FP32_error']['failed'] and bool(not bad.any())==mode['ideal_FP32_error']['passed']
        assert mode['CPU_BF16_P_emulation_error']['passed'] and mode['cache_and_guards_RNG_preserved']
        exact=torch.equal(actual,a['outputs']['control']);assert exact==mode['bitwise_vs_control'];checks+=1
        if not mode['ideal_FP32_error']['passed']:ideal_failed_modes.append(dict(case=i,label=mode['label'],failed=int(bad.sum()),max_abs=mode['ideal_FP32_error']['max_abs']))
        if ql==kl:assert exact;exact_control_equivalence.append(dict(case=i,label=mode['label']))
original=Path('/workspace/benchmark-results/20261021_multiquery_splitkv_GPU_tianshu/result.json')
assert json.loads(original.read_text())['phase']=='failed'
result=dict(complete=True,CPU_reverified=True,diagnostic_only=True,model_executed=False,benchmark=False,
    original_26_case_plan_still_failed=True,original_failed_result_sha256=sha(original),raw_result_sha256=sha(ROOT/'result.json'),raw_root=str(ROOT),
    original_FP32_tolerance_unchanged=True,all_24_modes_match_native_BF16_P_emulation=True,CPU_recomputed_modes=checks,
    ideal_FP32_failed_modes=ideal_failed_modes,short_q_equals_k_bitwise_control_equivalence=exact_control_equivalence,
    full_observations=r['cases'],frozen_sources=r['frozen_sources'],verifier_source_sha256=sha(__file__))
(BASE/'DIAGNOSTIC_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('full_observations','frozen_sources','short_q_equals_k_bitwise_control_equivalence')}))
