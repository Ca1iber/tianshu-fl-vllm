# SPDX-License-Identifier: Apache-2.0
"""Compare the actual native block with integer formulas, including buffer tails."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from source import BASE, NATIVE, baseline_function, eligible, render

p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
root = args.output.resolve(); assert root.is_relative_to('/workspace/benchmark-results') and not root.exists(); root.mkdir(parents=True)


class Buffer:
    def __init__(self, length, dtype):
        self.cpu = torch.arange(length, dtype=dtype) + 701
        self.np = self.cpu.numpy()
        self.gpu = self.cpu.clone() + 1000
        self.copies = []
    def copy_to_gpu(self, count=None):
        if count is None: count = len(self.cpu)
        self.gpu[:count].copy_(self.cpu[:count]); self.copies.append(count)


def owner(capacity, phase):
    return NS(num_computed_tokens=(torch.arange(capacity, dtype=torch.int32) * 13 + phase * 4091) % 131072,
        req_indices=Buffer(4096, torch.int64), query_pos=Buffer(4096, torch.int64),
        num_scheduled_tokens=Buffer(capacity, torch.int32), positions=torch.arange(4096, dtype=torch.int64) - 100,
        seq_lens=torch.arange(capacity, dtype=torch.int32) + 100,
        uses_mrope=False, uses_xdrope_dim=0, speculative_config=None, enable_prompt_embeds=False,
        mamba_prev_last_scheduled_idx=None, use_async_spec_decode=False)


def reference(o, req, counts, n, total):
    o.req_indices.np[:total] = req
    o.num_scheduled_tokens.np[:n] = counts
    o.req_indices.gpu[:n] = torch.arange(n, dtype=torch.int64)
    o.query_pos.gpu[:n] = 0
    o.num_scheduled_tokens.gpu[:n] = 1
    o.positions[:n] = o.num_computed_tokens[:n].to(torch.int64)
    o.seq_lens[:n] = o.num_computed_tokens[:n] + 1
    o.seq_lens[n:] = 0

baseline = baseline_function(); rows = []
rng_before = torch.get_rng_state().clone()
for capacity in (64, 256):
    for n in range(1, 65):
        for phase in (0, 1, 31):
            a, b = owner(capacity, phase), owner(capacity, phase)
            # These CPU query positions were computed earlier by the unchanged native code.
            a.query_pos.np[:n] = 0; b.query_pos.np[:n] = 0
            indices = np.arange(n, dtype=np.int64); counts = np.ones(n, dtype=np.int32)
            before = a.num_computed_tokens.clone()
            baseline(a, indices, counts, n, n); reference(b, indices, counts, n, n)
            assert torch.equal(a.num_computed_tokens, before) and torch.equal(b.num_computed_tokens, before)
            for name in ('req_indices', 'query_pos', 'num_scheduled_tokens'):
                x, y = getattr(a, name), getattr(b, name)
                assert torch.equal(x.cpu, y.cpu) and torch.equal(x.gpu, y.gpu), (name, n, capacity)
            assert torch.equal(a.positions, b.positions) and torch.equal(a.seq_lens, b.seq_lens)
            assert a.req_indices.copies == a.query_pos.copies == a.num_scheduled_tokens.copies == [n]
            rows.append(dict(capacity=capacity, batch=n, phase=phase, passed=True))
assert torch.equal(torch.get_rng_state(), rng_before)
text, provenance = render(); (root / 'generated_prepare_inputs.py').write_text(text)
# Guard checks use descriptor-only CUDA tensors; no GPU allocation is performed.
class Meta:
    def __init__(self, tensor): self.dtype=tensor.dtype; self.device=torch.device('cuda:0'); self.ndim=tensor.ndim; self.length=tensor.numel(); self.pitch=1
    def numel(self): return self.length
    def stride(self, dim): return self.pitch

def metadata_owner():
    o = owner(256, 0)
    for name in ('num_computed_tokens', 'positions', 'seq_lens'): setattr(o, name, Meta(getattr(o, name)))
    for name in ('req_indices', 'query_pos', 'num_scheduled_tokens'): getattr(o, name).gpu = Meta(getattr(o, name).gpu)
    return o

guards=[]
o=metadata_owner(); assert eligible(o,np.ones(64,dtype=np.int32),64,64)
for name, value in [('uses_mrope',True),('uses_xdrope_dim',3),('speculative_config',object()),('enable_prompt_embeds',True),('mamba_prev_last_scheduled_idx',object()),('use_async_spec_decode',True)]:
    old=getattr(o,name);setattr(o,name,value);assert not eligible(o,np.ones(64,dtype=np.int32),64,64);setattr(o,name,old);guards.append(name)
for counts,n,total,label in [(np.ones(65,dtype=np.int32),65,65,'batch65'),(np.array([],dtype=np.int32),0,0,'empty'),(np.array([0,2]),2,2,'mixed'),(np.array([2,2]),2,4,'prefill'),(np.ones(2),2,3,'total_mismatch')]:
    assert not eligible(o,counts,n,total);guards.append(label)
o.positions.pitch=2;assert not eligible(o,np.ones(64),64,64);guards.append('strided_position');o.positions.pitch=1
o.req_indices.gpu.dtype=torch.int32;assert not eligible(o,np.ones(64),64,64);guards.append('wrong_index_dtype')
assert not eligible(owner(256,0),np.ones(64),64,64);guards.append('CPU_device')
paths=[BASE/'kernel.py',BASE/'source.py',BASE/'check_cpu.py',NATIVE]
gate=dict(complete=True,CPU_only=True,GPU_not_run=True,adopted=False,native_block_AST_executed=True,
    CPU_state_comparisons=len(rows),metadata_guard_checks=len(guards)+1,native_three_H2D_copies_observed=True,
    full_buffer_tails_and_input_immutability_verified=True,RNG_unchanged=True,provenance=provenance,
    source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    generated_source_sha256=hashlib.sha256(text.encode()).hexdigest(),guard_cases=guards)
(root/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps(dict(complete=True,CPU_state_comparisons=len(rows),metadata_guard_checks=len(guards)+1,provenance=provenance)))
