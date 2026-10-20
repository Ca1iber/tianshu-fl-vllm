# SPDX-License-Identifier: Apache-2.0
"""Recompute full real CPU packed-payload hashes and all archived synthetic values."""
import hashlib,json,statistics,sys
from pathlib import Path
import torch
from safetensors import safe_open
BASE=Path(__file__).resolve().parent;root=Path(sys.argv[1]).resolve();result=json.loads((root/'result.json').read_text())
assert result['complete'] and result['phase']=='complete' and len(result['banks'])==43 and len(result['synthetic'])==9
for path,sha in result['frozen_sources'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((root/'source_snapshot'/Path(path).relative_to('/')).read_bytes()).hexdigest()==sha
reference={r['prefix']:r for r in json.loads(Path(result['source_signature']).read_text())}
def tensor_sha(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
def pack(w):
    q=torch.empty(w.shape,dtype=torch.int8);s=torch.empty(len(w),dtype=torch.float32)
    for lo in range(0,len(w),8192):
        block=w[lo:lo+8192].float();scale=block.abs().amax(1).clamp_min(1.e-8)/127.
        q[lo:lo+len(block)]=torch.round(block/scale[:,None]).clamp(-127,127).to(torch.int8);s[lo:lo+len(block)]=scale
    return q,s
cpu_elements=0
files=list(Path('/workspace/MiniCPM5-2B').glob('*.safetensors'));assert len(files)==1
with safe_open(str(files[0]),framework='pt',device='cpu') as checkpoint:
    for bank in result['banks']:
        wc=torch.cat([checkpoint.get_tensor(key) for key in bank['checkpoint_keys']],0).contiguous();q,s=pack(wc)
        expected=dict(original_BF16_sha256=tensor_sha(wc),packed_sha256=tensor_sha(q),scales_sha256=tensor_sha(s))
        assert expected=={key:reference[bank['prefix']][key] for key in expected}
        assert len(bank['rounds'])==3
        for phase in bank['rounds']:
            assert phase['signature']==expected and all(phase[key] for key in ('full_CPU_GPU_q_and_scale_bitwise_equal','original_input_bitwise_unchanged','RNG_unchanged','guards_passed'))
        cpu_elements+=q.numel()+s.numel()
for case in result['synthetic']:
    data=torch.load(root/case['file'],map_location='cpu',weights_only=True);q,s=pack(data['input'])
    assert torch.equal(q,data['packed']) and torch.equal(s,data['scales'])
    assert all(case[key] for key in ('full_bitwise_equal','guards_passed','RNG_unchanged','input_bitwise_unchanged'))
    cpu_elements+=q.numel()+s.numel()
timings=result['timing_rounds'];assert len(timings)==129
for bank in result['banks']:
    rows=[row for row in timings if row['prefix']==bank['prefix']];assert len(rows)==3
    assert all(row['order']==(['native','candidate'] if row['round']%2==0 else ['candidate','native']) and row['native_ms']>0 and row['candidate_ms']>0 for row in rows)
native=sum(statistics.median(r['native_ms'] for r in timings if r['prefix']==b['prefix']) for b in result['banks'])
candidate=sum(statistics.median(r['candidate_ms'] for r in timings if r['prefix']==b['prefix']) for b in result['banks'])
gate=dict(complete=True,independently_CPU_rehashed_all_real_banks=True,real_banks=43,full_GPU_CPU_exact_comparisons=129,synthetic_changed_input_cases=9,
    CPU_recomputed_quantized_and_scale_elements=cpu_elements,all_43_payloads_match_fresh_Math_signature=True,CPU_threads=result['CPU_threads'],
    sum_of_bank_median_loader_style_ms=dict(native=native,candidate=candidate),candidate_latency_reduction_percent=(1-candidate/native)*100,
    timing_scope=result['timing_scope'],timing_not_actual_model_startup=True,model_executed=False,adopted=False,official_benchmark=False,
    checkpoint_never_modified=True,frozen_sources=result['frozen_sources'],raw_root=str(root))
(BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'GPU_ROUNDS_32g.json').write_text(json.dumps(timings,indent=2)+'\n')
print(json.dumps(gate))
