# SPDX-License-Identifier: Apache-2.0
"""V2 keeps unchanged exponential RNG eager; strict state/sequence checks retained."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    root.mkdir(parents=True)
    signal.alarm(900)
    result = dict(phase='preparing', complete=False, official_benchmark=False, adopted=False,
                  runner_pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                  cases=[], state_checks=[], guard_checks=[], timings=[], source_snapshots={})

    def save():
        (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: result.get(k) for k in ('phase','error')}, ensure_ascii=False), flush=True)

    def snapshot(path):
        path = Path(path).resolve()
        data = path.read_bytes()
        result['source_snapshots'][str(path)] = hashlib.sha256(data).hexdigest()
        dest = root/'source_snapshot'/path.relative_to('/')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    try:
        hardware = subprocess.check_output(['ixsmi'], text=True)
        (root/'hardware_before.txt').write_text(hardware)
        assert 'No running processes found' in hardware
        base = Path(__file__).resolve().parent
        repo = base.parents[1]
        env = dict(os.environ, TIANSHU_GEO_PROFILE_DRY_RUN='1', TIANSHU_GEO_RECORD_DIR=str(root/'profile'))
        launcher = repo/'experiments/tianshu_gemm_topp_geo_20261016/serve_tianshu_minicpm.sh'
        launch = json.loads(subprocess.check_output(['bash', str(launcher)], env=env, text=True))
        profile = launch['profile_environment']
        os.environ.update(profile)
        runpy.run_path(str(Path(profile['PYTHONPATH'])/'sitecustomize.py'))
        import torch
        import vllm_fl
        import flag_gems
        from types import SimpleNamespace
        from vllm.v1.sample.sampler import Sampler
        from vllm.v1.sample.metadata import SamplingMetadata
        from graph_sampler_v2 import SamplerGraph, eligible
        import graph_sampler_v2 as graph_sampler
        import inspect
        import vllm.v1.sample.ops.topk_topp_sampler as top_module
        torch.set_num_threads(8)
        template = json.loads((launcher.parent/'serve_profile_template.json').read_text())
        for path, sha in template['frozen_sources'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
            snapshot(path)
        for path in (Path(__file__), Path(graph_sampler.__file__), Path(inspect.getsourcefile(Sampler)),
                     Path(inspect.getsourcefile(SamplingMetadata)), Path(top_module.__file__), base/"graph_sampler.py",
                     Path(inspect.getsourcefile(flag_gems.exponential_))):
            snapshot(path)
        result.update(phase='capability', environment=profile, original_serve_argv=launch['command'])
        save()
        host_gen = torch.Generator().manual_seed(20261016)
        hosts = [torch.randn((64,130560), generator=host_gen, dtype=torch.float32).bfloat16() for _ in range(3)]
        hosts[1].mul_(5)
        hosts[2][:,4096:] = -float('inf')
        logits = [x.cuda() for x in hosts]
        blank = torch.empty(0, device='cuda')
        metadata = SamplingMetadata(temperature=torch.ones(64, device='cuda'), all_greedy=False,
            all_random=True, top_p=torch.full((64,),.95,device='cuda'), top_k=None, generators={},
            max_num_logprobs=None, no_penalties=True, prompt_token_ids=None,
            frequency_penalties=blank,presence_penalties=blank,repetition_penalties=blank,
            output_token_ids=[[] for _ in range(64)], allowed_token_ids_mask=None,bad_words_token_ids={},
            logitsprocs=SimpleNamespace(argmax_invariant=[],non_argmax_invariant=[]))
        sampler = Sampler()
        assert sampler.topk_topp_sampler.forward.__name__=='forward_native'
        original = type(sampler).forward
        with torch.inference_mode(), flag_gems.use_gems():
            assert eligible(sampler, logits[0], metadata)
            torch.cuda.manual_seed(12345)
            rng_before = torch.cuda.get_rng_state().clone()
            graph = SamplerGraph(sampler, original, logits[0], metadata)
            assert torch.equal(torch.cuda.get_rng_state(),rng_before)
            result['state_checks'].append(dict(name='warmup_capture_preserves_default_rng',passed=True))
            for index in range(9):
                scenario = index % 3
                temperature = (.7,1.,1.4)[scenario]
                top_p = (.8,.95,.99)[scenario]
                meta = replace(metadata,temperature=torch.full((64,),temperature,device='cuda'),
                    top_p=torch.full((64,),top_p,device='cuda'))
                # Different live allocations, changing logits and scalar vectors.
                x = logits[scenario].clone()
                torch.cuda.manual_seed(111+index)
                initial = torch.cuda.get_rng_state().clone()
                expected = original(sampler,x,meta)
                expected_state = torch.cuda.get_rng_state().clone()
                torch.cuda.set_rng_state(initial)
                actual = graph(x,meta)
                actual_state = torch.cuda.get_rng_state().clone()
                torch.cuda.synchronize()
                item = dict(index=index,distribution=('normal','peaked','masked')[scenario],temperature=temperature,
                    top_p=top_p,seed=111+index,expected=expected.sampled_token_ids.cpu().flatten().tolist(),
                    actual=actual.sampled_token_ids.cpu().flatten().tolist(),
                    rng_equal=torch.equal(expected_state,actual_state),
                    input_unchanged=torch.equal(x.cpu(),hosts[scenario]),
                    expected_rng_sha256=hashlib.sha256(expected_state.numpy().tobytes()).hexdigest(),
                    actual_rng_sha256=hashlib.sha256(actual_state.numpy().tobytes()).hexdigest())
                item['tokens_equal']=item['expected']==item['actual']
                result['cases'].append(item)
                save()
                assert item['tokens_equal'] and item['rng_equal'] and item['input_unchanged'],item
            torch.cuda.manual_seed(701)
            initial = torch.cuda.get_rng_state().clone()
            expected = [original(sampler,logits[i%3],metadata).sampled_token_ids.clone() for i in range(8)]
            expected_state = torch.cuda.get_rng_state().clone()
            torch.cuda.set_rng_state(initial)
            actual = [(graph(logits[i%3],metadata) if i%2==0 else original(sampler,logits[i%3],metadata)).sampled_token_ids for i in range(8)]
            torch.cuda.synchronize()
            mixed = dict(name='eight_mixed_eager_graph_draws_and_old_output_lifetime',
                outputs=[x.cpu().flatten().tolist() for x in actual],
                expected=[x.cpu().flatten().tolist() for x in expected],
                rng_equal=torch.equal(torch.cuda.get_rng_state(),expected_state),
                independent_outputs=len({x.data_ptr() for x in actual})==8)
            mixed['tokens_equal']=mixed['outputs']==mixed['expected']
            result['state_checks'].append(mixed)
            assert mixed['rng_equal'] and mixed['tokens_equal'] and mixed['independent_outputs'],mixed
            variants = dict(greedy=replace(metadata,all_random=False,all_greedy=True),
                penalties=replace(metadata,no_penalties=False),logprobs=replace(metadata,max_num_logprobs=0),
                specific_logprobs=replace(metadata,logprob_token_ids={0:[1]}),
                topk=replace(metadata,top_k=torch.full((64,),10,device='cuda',dtype=torch.int32)),
                generator=replace(metadata,generators={0:torch.Generator(device='cuda')}),
                badwords=replace(metadata,bad_words_token_ids={0:[[1]]}),
                holder=replace(metadata,thinking_budget_state_holder=object()),
                unknown_processor=replace(metadata,logitsprocs=SimpleNamespace(argmax_invariant=[object()],non_argmax_invariant=[])))
            for name,meta in variants.items():
                rejected=not eligible(sampler,logits[0],meta);result['guard_checks'].append(dict(name=name,rejected=rejected));assert rejected,name
            for name,x in [('FP32',logits[0].float()),('B8',logits[0][:8]),('vocab_short',logits[0][:,:1000])]:
                rejected=not eligible(sampler,x,metadata);result['guard_checks'].append(dict(name=name,rejected=rejected));assert rejected,name
            assert not eligible(sampler,logits[0],metadata,True)
            assert not eligible(sampler,logits[0],metadata,False,'raw_logits')
            funcs=[lambda:original(sampler,logits[0],metadata),lambda:graph(logits[0],metadata)]
            for f in funcs:
                for _ in range(5):f()
            result['phase']='timing';save()
            for repeat in range(7):
                for selected in ((0,1) if repeat%2==0 else (1,0)):
                    torch.cuda.synchronize();start=time.perf_counter()
                    ev0,ev1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    for _ in range(20):funcs[selected]()
                    ev1.record();ev1.synchronize()
                    result['timings'].append(dict(repeat=repeat,mode=('eager','graph')[selected],iterations=20,
                        host_wall_per_call_ms=1000*(time.perf_counter()-start)/20,
                        gpu_stream_per_call_ms=ev0.elapsed_time(ev1)/20))
            medians={mode:{metric:statistics.median(v[metric] for v in result['timings'] if v['mode']==mode)
                for metric in ('host_wall_per_call_ms','gpu_stream_per_call_ms')} for mode in ('eager','graph')}
            result.update(phase='complete',complete=True,timing_medians=medians,graph_replays=graph.replays,
                original_algorithm_and_global_rng_preserved=True,RNG_executed_outside_graph=True,scope='C64 staged full-Sampler micro; no model throughput claim')
        for path,sha in result['source_snapshots'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    except Exception as error:
        import traceback
        result.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        save()
    return 0 if result['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
