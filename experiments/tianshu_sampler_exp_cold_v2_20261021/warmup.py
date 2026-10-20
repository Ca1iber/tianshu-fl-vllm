# SPDX-License-Identifier: Apache-2.0
"""Prime the native RNG dispatch ABI classes without changing active entries."""
import hashlib,importlib,inspect,time
from pathlib import Path
NATIVE_SHA='2222bab5ecb28e66a7d9499c5611ca9dac73181cc094e0853fe3cdc5d0869728'
def tensor_sha(t):return hashlib.sha256(memoryview(t.cpu().contiguous().numpy())).hexdigest()
def prewarm(torch,operator,vocab=130560,maximum_batch=64,emit=print):
    assert vocab==130560 and maximum_batch==64 and torch.cuda.is_available()
    native=Path(inspect.getsourcefile(operator));assert hashlib.sha256(native.read_bytes()).hexdigest()==NATIVE_SHA
    module=importlib.import_module('flag_gems.ops.exponential_');entry=module.fused_exponential_kernel_f32_unroll8
    assert operator is module.exponential_ and entry.arg_names==['out_ptr','N','inv_lambd','eps_minus','philox_seed','philox_offset','BLOCK']
    device=torch.cuda.current_device();cache=entry.kernel_cache[device]
    cpu_before=torch.get_rng_state().clone();gpu_before=torch.cuda.get_rng_state(device).clone()
    default_seed=int(gpu_before.view(torch.int64)[0]);other_seed=2**31 if -(2**31)<=default_seed<2**31 else 0
    private=torch.Generator(device=f'cuda:{device}');scratch=torch.empty((64,vocab),device=device,dtype=torch.float32)
    eps=-.5*torch.finfo(torch.float32).eps;tasks=[]
    for tag in ('default','other'):
        if tag=='default':private.set_state(gpu_before)
        else:private.manual_seed(other_seed)
        seed=int(private.get_state().view(torch.int64)[0])
        for offset in (0,2**31):
            key=entry.key([scratch,1.,eps],[scratch.numel(),seed,offset],[])
            tasks.append((tag,seed,offset,key,cache.get(key)))
    assert len({t[3] for t in tasks})==4
    rows=[];start=time.perf_counter()
    for tag,seed,offset,key,old in tasks:
        if tag=='default':private.set_state(gpu_before)
        else:private.manual_seed(other_seed)
        private.set_offset(offset)
        emit(f'PRIME_RNG_ABI {tag} {key[-2:]}',flush=True)
        begin=time.perf_counter();returned=operator(scratch,generator=private);torch.cuda.synchronize(device)
        assert returned.data_ptr()==scratch.data_ptr() and key in cache
        if old is not None:assert cache[key] is old
        config={k:v for k,v in cache[key][1].items() if v is None or isinstance(v,(bool,int,float,str))}
        rows.append(dict(seed_group=tag,seed_state_word=seed,offset=offset,N=scratch.numel(),dns_classes=list(key[-3:]),
            entry_present_before=old is not None,preexisting_entry_preserved=old is None or cache[key] is old,
            elapsed_s=time.perf_counter()-begin,selected_config=config))
    assert torch.equal(cpu_before,torch.get_rng_state()) and torch.equal(gpu_before,torch.cuda.get_rng_state(device))
    elapsed=time.perf_counter()-start
    del scratch,private
    return dict(complete=True,calls=rows,representative_batch=64,vocab=vocab,dispatch_classes=4,
        actual_entry_presence_checked=True,all_preexisting_dispatch_entries_preserved=True,
        N_value_not_in_native_dispatch_key=True,CPU_RNG_unchanged=True,default_GPU_RNG_unchanged=True,
        CPU_RNG_sha256=tensor_sha(cpu_before),GPU_RNG_sha256=tensor_sha(gpu_before),elapsed_s=elapsed,allocation_bytes=64*vocab*4)
