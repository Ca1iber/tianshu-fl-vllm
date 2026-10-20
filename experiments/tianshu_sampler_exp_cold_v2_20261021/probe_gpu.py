# SPDX-License-Identifier: Apache-2.0
"""Controlled native cold-cache fixture; no library/database mutations survive."""
import argparse,hashlib,importlib,inspect,json,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import warmup
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config(c):return None if c is None else dict(c.all_kwargs())
class MemoryBench:
    def __init__(self):self.values={};self.misses=0;self.hits=0;self.writes=0
    def get(self,c):
        key=tuple(sorted(c.all_kwargs().items()));value=self.values.get(key)
        if value is None:self.misses+=1
        else:self.hits+=1
        return value
    def __setitem__(self,c,v):self.values[tuple(sorted(c.all_kwargs().items()))]=tuple(v);self.writes+=1
class BenchmarkProxy:
    def __init__(self,original,table):self.original=original;self.table=table;self.pools={};self.target_accesses=0
    def __getitem__(self,key):
        if isinstance(key,tuple) and len(key)==2 and key[0]==self.table:
            self.target_accesses+=1;return self.pools.setdefault(key[1],MemoryBench())
        return self.original[key]
    def __getattr__(self,name):return getattr(self.original,name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args()
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>600;signal.alarm(min(remaining,1200))
    state=dict(complete=False,phase='preparing',component_only=True,adopted=False,service_not_installed=True,Math_not_run=True)
    def save():(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    save();module=tuner=entry=old_cache=old_libcache=None
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
        import torch,flag_gems
        torch.set_num_threads(8)
        module=importlib.import_module('flag_gems.utils.libentry');native=importlib.import_module('flag_gems.ops.exponential_')
        operator=flag_gems.exponential_;assert operator is native.exponential_ and flag_gems.vendor_name=='iluvatar'
        entry=native.fused_exponential_kernel_f32_unroll8;tuner=entry.fn
        assert isinstance(tuner,module.LibTuner) and tuner._run_mode==module.LibTunerRunMode.NORMAL
        cpu_gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert cpu_gate['complete']
        for path,digest in cpu_gate['sources'].items():assert sha(path)==digest
        sources=dict(cpu_gate['sources']);sources[str(BASE/'CPU_GATE_32g.json')]=sha(BASE/'CPU_GATE_32g.json')
        for path,digest in sources.items():
            target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==digest
        torch.cuda.synchronize();device=torch.cuda.current_device();cpu_before=torch.get_rng_state().clone();gpu_before=torch.cuda.get_rng_state(device).clone()
        default_seed=int(gpu_before.view(torch.int64)[0]);vocab=130560;eps=-.5*torch.finfo(torch.float32).eps
        scratch=torch.empty((64,vocab),device=device,dtype=torch.float32);q=scratch[:37]
        def args_for(batch):return dict(out_ptr=scratch[:batch],N=batch*vocab,inv_lambd=1.,eps_minus=eps,philox_seed=default_seed,philox_offset=0)
        keys={str(b):tuner.get_key(args_for(b)) for b in (37,64)}
        old_cache=tuner.cache;old_libcache=module.libcache
        original_configs={b:config(old_cache.get(key)) for b,key in keys.items()}
        memory={};proxy=BenchmarkProxy(old_libcache,tuner.benchmark_table_name)
        tuner.cache=memory;module.libcache=proxy;module.clear_libentry_dispatch_cache(entry)
        state.update(phase='cold_boundary',source_snapshots=sources,fixture='Only target tuner ConfigCache and BenchmarkCache replaced by empty in-memory objects',
            original_configs_before=original_configs,cold_config_cache_initial_count=len(memory),cold_benchmark_initial_count=len(proxy.pools));save()
        private=torch.Generator(device=f'cuda:{device}');private.set_state(gpu_before);private.set_offset(2**31)
        print('CONTROL_COLD37_HIGH_BEGIN',flush=True)
        start=time.perf_counter();operator(q,generator=private);torch.cuda.synchronize();cold_s=time.perf_counter()-start
        high_before=q.clone();cold_cfg=config(next(iter(memory.values())))
        cold_success=tuner.benchmark_success_count;cold_hits=tuner.benchmark_cache_hit_count
        print('CONTROL_COLD37_HIGH_END',flush=True)
        assert len(memory)==1 and cold_success>0 and cold_hits==0
        module.clear_libentry_dispatch_cache(entry)
        private.set_state(gpu_before);operator(q,generator=private);torch.cuda.synchronize();default_before=q.clone()
        low_key=entry.key([q,1.,eps],[q.numel(),default_seed,int(gpu_before.view(torch.int64)[1])],[])
        old_low_entry=entry.kernel_cache[device][low_key]
        state.update(phase='prewarming',cold_boundary_elapsed_s=cold_s,cold_boundary_measured_configurations=cold_success,cold_boundary_benchmark_hits=cold_hits,cold_boundary_config=cold_cfg);save()
        print('PREWARM_ABI_BEGIN',flush=True);warm=warmup.prewarm(torch,operator);print('PREWARM_ABI_END',flush=True)
        assert entry.kernel_cache[device][low_key] is old_low_entry and len(memory)==2
        private.set_state(gpu_before);operator(q,generator=private);torch.cuda.synchronize();default_after=q.clone()
        assert torch.equal(default_before.view(torch.int32),default_after.view(torch.int32))
        private.set_state(gpu_before);private.set_offset(2**31)
        print('WARM_HIGH37_BEGIN',flush=True)
        start=time.perf_counter();operator(q,generator=private);torch.cuda.synchronize();warm_s=time.perf_counter()-start
        high_after=q.clone();print('WARM_HIGH37_END',flush=True)
        state['phase']='replay';save();print('REPLAY_ALL_CLASSES_BEGIN',flush=True)
        default_word=int(gpu_before.view(torch.int64)[0]);other_seed=2**31 if -(2**31)<=default_word<2**31 else 0;replay=[]
        for seed_group in ('default','other'):
            for offset in (0,2**31):
                for batch in range(1,65):
                    if seed_group=='default':private.set_state(gpu_before)
                    else:private.manual_seed(other_seed)
                    private.set_offset(offset);value=scratch[:batch]
                    operator(value,generator=private);torch.cuda.synchronize()
                    assert bool(torch.isfinite(value).all() and (value>0).all())
                    replay.append(dict(seed_group=seed_group,offset=offset,batch=batch))
        print('REPLAY_ALL_CLASSES_END',flush=True)
        assert len(entry.kernel_cache[device])==4 and torch.equal(cpu_before,torch.get_rng_state()) and torch.equal(gpu_before,torch.cuda.get_rng_state(device))
        torch.save(dict(default_before=default_before.cpu(),default_after=default_after.cpu(),high_before=high_before.cpu(),high_after=high_after.cpu(),CPU_before=cpu_before,CPU_after=torch.get_rng_state(),GPU_before=gpu_before,GPU_after=torch.cuda.get_rng_state(device)),root/'B37_and_RNG_fullwords.pt')
        state.update(prewarm=warm,replay=replay,actual_entry_classes=len(entry.kernel_cache[device]),actual_ConfigCache_keys=len(memory),
            memory_configurations=[dict(key=repr(k),configuration=config(v)) for k,v in memory.items()],
            cold_boundary_elapsed_s=cold_s,warm_boundary_elapsed_s=warm_s,
            benchmark_pools=[dict(key=repr(k),measured=len(v.values),misses=v.misses,hits=v.hits,writes=v.writes) for k,v in proxy.pools.items()],
            default_B37_words_equal=default_before.numel(),high_B37_different_words=int((high_before.view(torch.int32)!=high_after.view(torch.int32)).sum()),
            default_CPU_and_GPU_RNG_unchanged=True,existing_low_entry_preserved=True,
            native_run_mode_and_protocol_unchanged=tuner._run_mode==module.LibTunerRunMode.NORMAL)
        tuner.cache=old_cache;module.libcache=old_libcache;module.clear_libentry_dispatch_cache(entry)
        after={b:config(old_cache.get(key)) for b,key in keys.items()};assert after==original_configs
        assert all(sha(p)==digest for p,digest in sources.items())
        state.update(phase='complete',complete=True,original_cache_objects_restored=tuner.cache is old_cache and module.libcache is old_libcache,
            original_selected_37_and_64_configs_unchanged=True,original_configs_after=after)
        save();print(json.dumps({k:v for k,v in state.items() if k not in ('source_snapshots','replay','prewarm')}),flush=True)
    except BaseException as error:
        import traceback
        state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
    finally:
        if tuner is not None and old_cache is not None:tuner.cache=old_cache
        if module is not None and old_libcache is not None:module.libcache=old_libcache
        if module is not None and entry is not None:module.clear_libentry_dispatch_cache(entry)
        signal.alarm(0)
if __name__=='__main__':main()
