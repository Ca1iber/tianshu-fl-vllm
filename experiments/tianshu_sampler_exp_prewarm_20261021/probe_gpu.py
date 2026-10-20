# SPDX-License-Identifier: Apache-2.0
"""Actual native cache/RNG preservation probe, not a model benchmark."""
import argparse,hashlib,importlib,inspect,json,os,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import warmup
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>900;signal.alarm(min(remaining,1200))
    state=dict(complete=False,phase='preparing',adopted=False,component_only=True,Math_not_run=True,service_not_installed=True)
    def save():(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
        import torch,flag_gems
        torch.set_num_threads(8)
        operator=flag_gems.exponential_;native=Path(inspect.getsourcefile(operator)).resolve()
        assert native==Path('/usr/local/lib/python3.12/site-packages/flag_gems/ops/exponential_.py') and flag_gems.vendor_name=='iluvatar'
        random_module=importlib.import_module('flag_gems.utils.random_utils')
        paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),native,Path(random_module.__file__),Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_sampler.py')]
        sources={str(p):sha(p) for p in paths}
        for p,digest in sources.items():
            target=root/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes());assert sha(target)==digest
        state.update(source_snapshots=sources,native_operator_source=str(native),native_operator_sha256=sha(native),phase='baseline');save()
        torch.cuda.synchronize();device=torch.cuda.current_device()
        cpu_initial=torch.get_rng_state().clone();gpu_initial=torch.cuda.get_rng_state(device).clone()
        print('BASELINE_37_BEGIN',flush=True)
        with torch.random.fork_rng(devices=[device]):
            baseline=torch.empty((37,130560),device=device,dtype=torch.float32);operator(baseline);torch.cuda.synchronize()
        assert torch.equal(cpu_initial,torch.get_rng_state()) and torch.equal(gpu_initial,torch.cuda.get_rng_state(device))
        print('BASELINE_37_END',flush=True)
        state['phase']='warming';save();print('PREWARM_BEGIN',flush=True)
        result=warmup.prewarm(torch,operator);state['prewarm']=result;save()
        print('PREWARM_END',flush=True)
        with torch.random.fork_rng(devices=[device]):
            actual=torch.empty_like(baseline);operator(actual);torch.cuda.synchronize()
        assert torch.equal(baseline.view(torch.int32),actual.view(torch.int32))
        assert bool(torch.isfinite(actual).all() and (actual>0).all())
        assert torch.equal(cpu_initial,torch.get_rng_state()) and torch.equal(gpu_initial,torch.cuda.get_rng_state(device))
        baseline_choices=baseline.reciprocal().argmax(-1);actual_choices=actual.reciprocal().argmax(-1)
        assert torch.equal(baseline_choices,actual_choices)
        private=torch.Generator(device=f'cuda:{device}');private.set_state(gpu_initial)
        replay=[];scratch=torch.empty((64,130560),device=device,dtype=torch.float32)
        print('REPLAY_SWEEP_BEGIN',flush=True)
        for batch in range(64,0,-1):
            value=scratch[:batch];start=time.perf_counter();operator(value,generator=private);torch.cuda.synchronize()
            assert bool(torch.isfinite(value).all() and (value>0).all())
            replay.append(dict(batch=batch,elapsed_s=time.perf_counter()-start))
        print('REPLAY_SWEEP_END',flush=True)
        assert torch.equal(cpu_initial,torch.get_rng_state()) and torch.equal(gpu_initial,torch.cuda.get_rng_state(device))
        torch.save(dict(baseline=baseline.cpu(),actual=actual.cpu(),baseline_choices=baseline_choices.cpu(),actual_choices=actual_choices.cpu(),CPU_before=cpu_initial,CPU_after=torch.get_rng_state(),GPU_before=gpu_initial,GPU_after=torch.cuda.get_rng_state(device)),root/'RNG_and_B37_fullwords.pt')
        assert all(sha(p)==digest for p,digest in sources.items())
        state.update(complete=True,phase='complete',B37_FP32_words_exact=actual.numel(),B37_choices_exact=37,
            CPU_RNG_unchanged=True,GPU_RNG_unchanged=True,replay=replay,
            output_identity_scope='Same already-selected B37 native tuning configuration, same RNG state; other seeds/configuration mapping not claimed')
        save();print(json.dumps({k:v for k,v in state.items() if k not in ('prewarm','source_snapshots','replay')}),flush=True)
    except BaseException as error:
        import traceback
        state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
