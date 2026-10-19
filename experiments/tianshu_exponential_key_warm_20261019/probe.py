# SPDX-License-Identifier: Apache-2.0
"""Use a private cold tuning database, then check unchanged native sampling/RNG."""
import argparse,hashlib,importlib,inspect,json,os
from datetime import datetime,timezone
from pathlib import Path
import signal,subprocess,sys
from warm import warm_exponential


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);a=p.parse_args()
    root=a.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
    remaining=int((datetime.fromisoformat(a.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200
    signal.alarm(min(remaining,900))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
    assert 'flag_gems' not in sys.modules
    cache=root/'isolated_flag_cache';database=root/'isolated_tuning.sqlite';assert not cache.exists() and not database.exists()
    os.environ.update(FLAGGEMS_CACHE_DIR=str(cache),FLAGGEMS_DB_URL='sqlite:///'+str(database),TRITON_PRINT_AUTOTUNING='1',VLLM_PLUGINS='fl')
    import torch
    import vllm_fl
    import flag_gems
    import vllm.v1.sample.ops.topk_topp_sampler as native
    import flag_gems.utils.random_utils as random_utils
    from warm import warm_exponential
    module=importlib.import_module('flag_gems.ops.exponential_')
    base=Path(__file__).resolve().parent
    gate=json.loads((base/'CPU_GATE_32g.json').read_text());assert gate['complete']
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for n,h in gate['sources'].items():assert digest(base/n)==h
    assert digest(native.__file__)=='35b9a1ffeb1e50b73b0ddd1ffc0f968910ee21cf9cc5455dd879fde547b6ddc1'
    assert inspect.getsourcefile(flag_gems.exponential_)==module.__file__
    sources={}
    paths=[base/n for n in ('warm.py','probe.py','check_cpu.py','run_probe.py','verify_downloaded.py','CPU_GATE_32g.json')]
    paths += [Path(native.__file__),Path(module.__file__),Path(inspect.getsourcefile(torch.random.fork_rng))]
    paths += [Path(inspect.getsourcefile(importlib.import_module('flag_gems.utils.'+n))) for n in ('random_utils','libentry','code_cache')]
    for p in paths:
        sources[str(p)]=digest(p);dest=root/'source_snapshot'/p.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    data=dict(phase='preparing',complete=False,official_benchmark=False,adopted=False,source_snapshots=sources,
        isolated_cold_tuning_database=str(database),isolated_cache=str(cache),shared_FlagGems_tuning_DB_not_used=True,isolated_FlagGems_config_and_code_cache=True,
        existing_compiler_cache_not_cleared=True,producer_records=[],sampling_cases=[],warm_calls=[],
        no_model_or_new_Math_claim=True)
    def save():(root/'result.json').write_text(json.dumps(data,indent=2)+'\n')
    phase=['cold_warm'];original=module.fused_exponential_kernel_f32_unroll8
    assert not original._has_flagtune_tuner
    class Trace:
        def __getitem__(self,grid):
            launch=original[grid]
            def run(*args,**kwargs):
                assert len(args)==6 and not kwargs and args[0].dtype==torch.float32 and args[0].is_contiguous()
                n=int(args[1]);assert n%130560==0 and 1<=n//130560<=64
                result=launch(*args,**kwargs)
                assert isinstance(result,tuple) and len(result)==2
                kernel,constants=result
                data['producer_records'].append(dict(phase=phase[0],N=n,batch=n//130560,
                    BLOCK=constants['BLOCK'],warps=kernel.metadata.num_warps,stages=kernel.metadata.num_stages,kernel_hash=kernel.hash,
                    philox_seed=int(args[4]),philox_offset=int(args[5]),
                    entry_key=repr(original.key([args[i] for i in original.specialize_indices if i<len(args)],
                        [args[i] for i in original.do_not_specialize_indices if i<len(args)],[]))))
                return result
            return run
    module.fused_exponential_kernel_f32_unroll8=Trace()
    def state():
        return torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
    def same(a,b):return all(torch.equal(x,y) for x,y in zip(a,b))
    def encoded(s):return dict(cpu=s[0].cpu().numpy().tobytes().hex(),cuda=s[1].cpu().numpy().tobytes().hex())
    save()
    try:
        torch.set_num_threads(8)
        with torch.inference_mode(),flag_gems.use_gems():
            initial=state();print('COLD_WARM_BEGIN',flush=True)
            data['warm_calls'].append(warm_exponential(torch,random_utils.set_philox_state))
            assert same(initial,state())
            lib=importlib.import_module('flag_gems.utils.libentry')
            assert lib.LibCache._instance is not None
            observed_url=lib.LibCache._instance.db_url
            assert observed_url==os.environ['FLAGGEMS_DB_URL']=='sqlite:///'+str(database)
            observed_cache=importlib.import_module('flag_gems.utils.code_cache').cache_dir_path()
            assert observed_cache==cache
            data.update(observed_libcache_db_url=observed_url,observed_FlagGems_cache_dir=str(observed_cache))
            print('COLD_WARM_END',flush=True)
            data.update(phase='postwarm',cold_warm_rng_before=encoded(initial),cold_warm_rng_after=encoded(state()));save()
            phase[0]='postwarm';print('POSTWARM_BEGIN',flush=True)
            bank=torch.empty((64,130560),dtype=torch.float32,device='cuda')
            order=list(range(1,65,2))+list(range(64,0,-2))
            for n in order:bank[:n].exponential_()
            torch.cuda.synchronize();del bank
            data['postwarm_batch_order']=order
            phase[0]='failure_warm'
            failure=RuntimeError('real warm fault sentinel')
            before_failure=state()
            attempts=[0]
            def fault_setter(seed,offset,device):
                attempts[0]+=1
                if attempts[0]==3:raise failure
                random_utils.set_philox_state(seed,offset,device)
            try:warm_exponential(torch,fault_setter)
            except RuntimeError as error:assert error is failure
            else:raise AssertionError('Fault swallowed')
            assert same(before_failure,state())
            data['real_failure_rng']=dict(before=encoded(before_failure),after=encoded(state()),original_exception_identity_preserved=True)
            for n in (1,2,3,7,8,16,24,32,62,63,64):
                hostgen=torch.Generator(device='cpu').manual_seed(2026101900+n)
                probs=torch.randn((n,130560),generator=hostgen,dtype=torch.float32).cuda().softmax(-1)
                for mode in ('global','mixed','all_seeded'):
                    ids=[] if mode=='global' else sorted({0,n-1}) if mode=='mixed' else list(range(n))
                    gens={i:torch.Generator(device='cuda').manual_seed(9000+i) for i in ids}
                    torch.cuda.manual_seed(2026101900+n)
                    initial=state();gs={i:g.get_state().clone() for i,g in gens.items()}
                    phase[0]='rng_reference'
                    expected=native.random_sample(probs.clone(),gens,False);expected_state=state();expected_gs={i:g.get_state().clone() for i,g in gens.items()}
                    torch.set_rng_state(initial[0]);torch.cuda.set_rng_state(initial[1])
                    for i,g in gens.items():g.set_state(gs[i])
                    phase[0]='repeat_warm';data['warm_calls'].append(warm_exponential(torch,random_utils.set_philox_state))
                    assert same(initial,state()) and all(torch.equal(g.get_state(),gs[i]) for i,g in gens.items())
                    after_warm=encoded(state());after_warm_gs={str(i):g.get_state().cpu().numpy().tobytes().hex() for i,g in gens.items()}
                    phase[0]='rng_actual';actual=native.random_sample(probs.clone(),gens,False)
                    assert torch.equal(actual,expected) and same(state(),expected_state)
                    assert all(torch.equal(g.get_state(),expected_gs[i]) for i,g in gens.items())
                    data['sampling_cases'].append(dict(batch=n,generator_mode=mode,seeded_rows=ids,
                        expected_tokens=expected.cpu().tolist(),actual_tokens=actual.cpu().tolist(),
                        initial_default=encoded(initial),after_warm_default=after_warm,after_warm_per_request=after_warm_gs,expected_final_default=encoded(expected_state),actual_final_default=encoded(state()),
                        initial_per_request={str(i):gs[i].cpu().numpy().tobytes().hex() for i in ids},
                        expected_final_per_request={str(i):expected_gs[i].cpu().numpy().tobytes().hex() for i in ids},
                        actual_final_per_request={str(i):gens[i].get_state().cpu().numpy().tobytes().hex() for i in ids},
                        warm_preserved_default_and_request_generators=True))
                    save()
                del probs
            data['scalar_boundary_cases']=[]
            for seed in (7,2**32+7):
                for offset in (0,2**31-4,2**31,2**32):
                    for n in (1,8,64):
                        hostgen=torch.Generator(device='cpu').manual_seed(321+n)
                        probs=torch.randn((n,130560),generator=hostgen,dtype=torch.float32).cuda().softmax(-1)
                        random_utils.set_philox_state(seed,offset,0)
                        initial=state();phase[0]='boundary_reference'
                        expected=native.random_sample(probs.clone(),{},False);expected_state=state()
                        torch.set_rng_state(initial[0]);torch.cuda.set_rng_state(initial[1])
                        phase[0]='boundary_repeat_warm';data['warm_calls'].append(warm_exponential(torch,random_utils.set_philox_state))
                        assert same(initial,state())
                        after_warm=encoded(state())
                        phase[0]='boundary_actual';actual=native.random_sample(probs.clone(),{},False)
                        assert torch.equal(actual,expected) and same(state(),expected_state)
                        data['scalar_boundary_cases'].append(dict(seed=seed,offset=offset,batch=n,
                            expected_tokens=expected.cpu().tolist(),actual_tokens=actual.cpu().tolist(),
                            initial_default=encoded(initial),after_warm_default=after_warm,
                            expected_final_default=encoded(expected_state),actual_final_default=encoded(state())))
                        save();del probs
            print('POSTWARM_END',flush=True)
        assert len(data['sampling_cases'])==33
        cold=[r for r in data['producer_records'] if r['phase']=='cold_warm']
        assert len(cold)==4 and len({r['entry_key'] for r in cold})==4 and {r['N'] for r in cold}=={8355840}
        assert len(data['scalar_boundary_cases'])==24
        for p,h in sources.items():assert digest(p)==h
        assert database.exists()
        data.update(phase='complete',complete=True,real_Torch_FlagGems_sampling_and_rng_checked=True)
    except BaseException as error:
        import traceback
        data.update(phase='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        module.fused_exponential_kernel_f32_unroll8=original;save();signal.alarm(0)


if __name__=='__main__':main()
