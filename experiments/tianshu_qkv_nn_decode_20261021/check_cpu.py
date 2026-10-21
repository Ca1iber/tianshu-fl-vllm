# SPDX-License-Identifier: Apache-2.0
"""Actual checkpoint, independent packing and static source gate; no CUDA."""
import ast,hashlib,json,os,subprocess
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',VLLM_PLUGINS='',PYTHONPATH='')
import numpy as np
import torch
from safetensors import safe_open
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
assert not torch.cuda.is_initialized();torch.set_num_threads(8)
with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as ck:
 w=torch.cat([ck.get_tensor('model.layers.0.self_attn.'+k+'_proj.weight') for k in ('q','k','v')]).contiguous()
assert tuple(w.shape)==(2560,2048)
v=w.float();scale=v.abs().amax(1).clamp_min(1.e-8)/127.;q=torch.round(v/scale[:,None]).clamp(-127,127).to(torch.int8)
a=v.numpy();s=np.maximum(np.max(np.abs(a),1),np.float32(1.e-8))/np.float32(127);nq=np.clip(np.rint(a/s[:,None]),-127,127).astype(np.int8)
assert np.array_equal(nq,q.numpy()) and np.array_equal(s,scale.numpy()) and torch.equal(q.T.contiguous().T,q)
sig=dict(prefix='model.layers.0.self_attn.qkv_proj',original_BF16_sha256=thash(w),packed_sha256=thash(q),scales_sha256=thash(scale),shape=list(w.shape),single_bank_payload_bytes=q.numel()+scale.numel()*4,potential42_bank_payload_bytes=42*(q.numel()+scale.numel()*4))
tiles=json.loads((BASE/'tiles.json').read_text());assert len(tiles)==17 and len({tuple(sorted(x.items())) for x in tiles})==17
for cfg in tiles:assert 2048%cfg['bk']==0 and 2560%cfg['bn']==0 and cfg['stages'] in (1,2)
for p in BASE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
subprocess.run(['bash','-n',str(BASE/'run_after_bf16.sh')],check=True)
files=[*BASE.glob('*.py'),BASE/'run_after_bf16.sh',BASE/'tiles.json',BASE/'PLAN_32g.md',EXP/'tianshu_dense_w8a8_20261020/kernel.py',EXP/'tianshu_dense_w8a8_20261020/GPU_GATE_32g.json',EXP/'tianshu_int8_weight_nn_20261021/kernel_nn.py',Path('/workspace/MiniCPM5-2B/config.json'),Path('/usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar/ops/mm.py'),Path('/usr/local/lib/python3.12/site-packages/flag_gems/utils/libentry.py')]
assert not torch.cuda.is_initialized()
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,rows=[8,64],tile_count=17,expected_cases=34,expected_dynamic_phases=102,INT32_max_absolute_dot_bound=2048*127*127,INT32_no_overflow=2048*127*127<2**31,weight_signature=sig,full_checkpoint_quantization_independent_NumPy_equal=True,source_count=len(files),sources={str(p):sha(p) for p in files},GPU_numerical_and_performance_gate_still_required=True)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
