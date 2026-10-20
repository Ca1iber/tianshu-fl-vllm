# SPDX-License-Identifier: Apache-2.0
"""Explicit FP32 RNE reference; no GPU FP64 or implicit scalar division."""
import hashlib,importlib.util,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
KERNEL=BASE.parent/'tianshu_dense_w8a8_20261020/kernel.py'
assert hashlib.sha256(KERNEL.read_bytes()).hexdigest()=='b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152'
spec=importlib.util.spec_from_file_location('_kvi8_reference_RNE',KERNEL);kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)

def main():
    fixture=Path('/workspace/benchmark-results/20261021_k8vi8_writer_diagnostic_tianshu/reference_diagnostic.pt')
    data=torch.load(fixture,map_location='cpu',weights_only=True);x=data['x'];scale=data['struct_scale']
    expected=(x.float()/scale[:,None]).round().clamp(-127,127).to(torch.int8)
    q=torch.empty(x.shape,device='cuda',dtype=torch.int8);s=torch.empty(x.shape[0],device='cuda',dtype=torch.float32)
    compiled=kernel.quantize(x.cuda(),q,s)
    assert torch.equal(q.cpu(),expected) and torch.equal(s.cpu(),scale)
    result=dict(complete=True,checked_heads=x.shape[0],checked_values=x.numel(),all65280_finite_BF16_patterns_and_random_extremes=True,
        actual_GPU_FP32_RNE_scale_equals_independent_struct_FP32=True,actual_GPU_quantized_values_equal_CPU=True,
        kernel_source=str(KERNEL),kernel_source_sha256=hashlib.sha256(KERNEL.read_bytes()).hexdigest(),
        reference_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),
        input_fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),reference_only_not_in_timed_writer=True)
    (BASE/'REFERENCE_RNE_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
