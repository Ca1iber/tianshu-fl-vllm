# SPDX-License-Identifier: Apache-2.0
"""Warm unchanged native exponential kernels with a dedicated CUDA generator."""
import hashlib,time

def tensor_sha(tensor):return hashlib.sha256(memoryview(tensor.cpu().contiguous().numpy())).hexdigest()
def prewarm(torch,operator,vocab=130560,maximum_batch=64,emit=print):
    assert vocab==130560 and maximum_batch==64 and torch.cuda.is_available()
    device=torch.cuda.current_device();cpu_before=torch.get_rng_state().clone();gpu_before=torch.cuda.get_rng_state(device).clone()
    private=torch.Generator(device=f'cuda:{device}');private.set_state(gpu_before)
    scratch=torch.empty((maximum_batch,vocab),device=device,dtype=torch.float32)
    rows=[];start=time.perf_counter()
    for offset_base in (0,2**31,2**32):
        private.set_offset(offset_base)
        emit(f'WARM_OFFSET_BEGIN {offset_base}',flush=True)
        for batch in range(1,maximum_batch+1):
            value=scratch[:batch];begin=time.perf_counter();returned=operator(value,generator=private);torch.cuda.synchronize(device)
            assert returned.data_ptr()==value.data_ptr()
            rows.append(dict(offset_base=offset_base,batch=batch,numel=value.numel(),elapsed_s=time.perf_counter()-begin))
        emit(f'WARM_OFFSET_END {offset_base}',flush=True)
    elapsed=time.perf_counter()-start
    cpu_after=torch.get_rng_state();gpu_after=torch.cuda.get_rng_state(device)
    assert torch.equal(cpu_before,cpu_after) and torch.equal(gpu_before,gpu_after)
    del scratch,private
    return dict(complete=True,batches=list(range(1,65)),vocab=vocab,offset_bases=[0,2**31,2**32],calls=rows,elapsed_s=elapsed,
        dedicated_generator_only=True,CPU_RNG_unchanged=True,default_GPU_RNG_unchanged=True,
        CPU_RNG_sha256=tensor_sha(cpu_before),GPU_RNG_sha256=tensor_sha(gpu_before),
        allocation_bytes=64*vocab*4,native_operator_unchanged=True)
