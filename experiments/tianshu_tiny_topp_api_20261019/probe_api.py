# SPDX-License-Identifier: Apache-2.0
"""Real mask API, changed-input CUDA Graphs and actual sampler RNG checks."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import statistics
import subprocess
import sys
import textwrap
from types import SimpleNamespace

from candidate_profile import prepare, digest, frozen_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-at-utc', required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    root.mkdir(parents=True)
    remaining = int((datetime.fromisoformat(args.stop_at_utc) - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 300
    result = dict(phase='preparing', runner_pid=os.getpid(), output_root=str(root), complete=False,
                  source_snapshots={}, masks=[], RNG_cases=[], fallbacks=[], events=[],
                  official_benchmark=False, model_test=False, fresh_Math=False, adopted=False)

    def save():
        temporary = root / 'result.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(root / 'result.json')
        print(json.dumps(dict(phase=result['phase'], masks=len(result['masks']),
                              RNG_cases=len(result['RNG_cases']), error=result.get('error'))), flush=True)

    def stop(signum, frame):
        raise TimeoutError(f'Stop requested: {signum}')

    def snapshot(path):
        path = Path(path).resolve()
        result['source_snapshots'][str(path)] = digest(path)
        destination = root / 'source_snapshot' / path.relative_to('/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.alarm(min(1800, remaining))
    save()
    try:
        hardware = subprocess.check_output(['ixsmi'], text=True)
        (root / 'hardware_before.txt').write_text(hardware)
        assert 'No running processes found' in hardware
        _, env, profile, sources = prepare(root, 'candidate')
        os.environ.clear()
        os.environ.update(env)
        runpy.run_path(str(Path(profile['PYTHONPATH']) / 'sitecustomize.py'))
        import torch
        import flag_gems
        import vllm_fl
        from vllm.v1.sample.ops import topk_topp_sampler as sampler
        from vllm.v1.sample import rejection_sampler
        assert sampler.apply_top_k_top_p._tianshu_tiny_candidate
        assert rejection_sampler.apply_top_k_top_p is sampler.apply_top_k_top_p
        candidate, original = sampler.apply_top_k_top_p, sampler._tianshu_tiny_original
        assert Path(sampler.TopKTopPSampler.forward_native.__code__.co_filename).resolve() == Path(sampler.__file__).resolve()
        assert Path(sampler.random_sample.__code__.co_filename).resolve() == Path(sampler.__file__).resolve()
        geo = sys.modules['_tianshu_topp_geometric_geo']
        fixture = Path(__file__).resolve().parents[1] / 'tianshu_topp_geometric_20261016/probe_precision.py'
        assert digest(fixture) == '3354e14c55aea477ba39497cf5f0378aadc83529f5d7956ceacb6197e9509d2c'
        namespace = dict(torch=torch)
        source = fixture.read_text()
        tree = ast.parse(source)
        for name in ('oracle', 'assess'):
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            body = textwrap.dedent('\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]))
            exec(compile(body, str(fixture), 'exec'), namespace)
        oracle, assess = namespace['oracle'], namespace['assess']
        torch.set_num_threads(8)
        for path in sources:
            snapshot(path)
        for path in (fixture, sampler.__file__, rejection_sampler.__file__, geo.__file__, flag_gems.__file__):
            snapshot(path)
        result.update(phase='mask_checks', imported_rejection_alias_verified=True,
                      profile_environment=profile, profile_frozen_sources=sources,
                      sampler_forward_unchanged=True)
        save()
        with torch.inference_mode(), flag_gems.use_gems():
            distributions = [(batch, 'normal') for batch in range(1, 8)] + [
                (1, 'peaked'), (2, 'bf16_peaked'), (4, 'ties'), (7, 'equal'),
                (3, 'masked'), (5, 'one_finite')]
            for case, (batch, distribution) in enumerate(distributions):
                torch.manual_seed(2026101900 + case)
                values = torch.randn(batch, 130560, device='cuda')
                if 'peaked' in distribution:
                    values.mul_(4)
                if distribution.startswith('bf16'):
                    values = values.to(torch.bfloat16).float()
                if distribution == 'ties':
                    values = values.round()
                if distribution == 'equal':
                    values.zero_()
                if distribution == 'masked':
                    values[:, ::3] = -float('inf')
                if distribution == 'one_finite':
                    values.fill_(-float('inf'))
                    values[:, 7] = 0
                for probability in (.0001, .1, .5, .95, .999, 1.0):
                    p = torch.full((batch,), probability, device='cuda')
                    working = {name: values.clone() for name in ('original', 'candidate')}
                    graphs = {}
                    for name, function in (('original', original), ('candidate', candidate)):
                        for _ in range(3):
                            function(working[name].copy_(values), None, p)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            working[name].copy_(values)
                            function(working[name], None, p)
                        graphs[name] = graph
                    for phase in (0, 1):
                        if phase:
                            values.copy_(values.roll(17, dims=1))
                        reference = oracle(values, None, p)
                        expected = original(values.clone(), None, p)
                        for name in working:
                            working[name].fill_(123)
                            graphs[name].replay()
                        torch.cuda.synchronize()
                        checks = {name: assess(working[name], expected, values, reference, require_strict=True)
                                  for name in working}
                        assert checks['candidate']['relative_gate'], (case, probability, phase, checks)
                        # Native precision at extreme p is recorded separately;
                        # the candidate must satisfy the strict CPU reference.
                        row = dict(case=case, batch=batch, distribution=distribution, probability=probability,
                                   phase=phase, cleared_outputs=True, changed_input=bool(phase), checks=checks)
                        if probability == .95:
                            times = {name: [] for name in working}
                            for trial in range(5):
                                for name in (('original', 'candidate') if trial % 2 == 0 else ('candidate', 'original')):
                                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                    start.record()
                                    for _ in range(10):
                                        graphs[name].replay()
                                    end.record()
                                    end.synchronize()
                                    ms = start.elapsed_time(end) / 10
                                    times[name].append(ms)
                                    result['events'].append(dict(case=case, phase=phase, mode=name, trial=trial, ms=ms))
                            row['medians_ms'] = {name: statistics.median(values) for name, values in times.items()}
                        result['masks'].append(row)
                    save()

            result['phase'] = 'actual_sampler_RNG'
            save()
            forward = sampler.TopKTopPSampler.forward_native
            for batch in range(1, 8):
                torch.manual_seed(2026101990 + batch)
                logits = torch.randn(batch, 130560, device='cuda')
                p = torch.full((batch,), .95, device='cuda')
                for seeded in ('none', 'partial', 'all'):
                    global_initial = torch.cuda.get_rng_state().clone()
                    count = 0 if seeded == 'none' else batch - 1 if seeded == 'partial' else batch
                    observed = {}
                    for name, function in (('original', original), ('candidate', candidate)):
                        generators = {i: torch.Generator(device='cuda').manual_seed(2026101970 + i) for i in range(count)}
                        torch.cuda.set_rng_state(global_initial)
                        sampler.apply_top_k_top_p = function
                        tokens, returned = forward(SimpleNamespace(logprobs_mode='processed_logits', use_fp64_gumbel=False),
                                                   logits.clone(), generators, None, p)
                        torch.cuda.synchronize()
                        observed[name] = dict(tokens=tokens.cpu(), mask=returned.cpu(),
                                              global_state=torch.cuda.get_rng_state().clone(),
                                              request_states={i: g.get_state().clone() for i, g in generators.items()})
                    sampler.apply_top_k_top_p = candidate
                    a, b = observed['original'], observed['candidate']
                    assert torch.equal(a['global_state'], b['global_state'])
                    assert all(torch.equal(a['request_states'][i], b['request_states'][i]) for i in range(count))
                    identical_mask = torch.equal(a['mask'], b['mask'])
                    identical_tokens = torch.equal(a['tokens'], b['tokens'])
                    if identical_mask:
                        assert identical_tokens
                    assert all(bool(torch.isfinite(b['mask'][i, int(token)])) for i, token in enumerate(b['tokens']))
                    state_hash = lambda tensor: hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
                    result['RNG_cases'].append(dict(batch=batch, seeded=seeded, global_and_request_states_identical=True,
                                                    identical_mask=identical_mask, identical_tokens=identical_tokens,
                                                    global_initial_sha256=state_hash(global_initial),
                                                    global_after_sha256={name: state_hash(value['global_state']) for name, value in observed.items()},
                                                    request_after_sha256={name: {str(i): state_hash(state) for i, state in value['request_states'].items()} for name, value in observed.items()},
                                                    sampled_tokens={name: value['tokens'].tolist() for name, value in observed.items()},
                                                    sampled_from_retained_logits=True))
            # Native TopK, no-filter and B>=8 fallback behavior must remain exact.
            for batch, topk, topp in ((4, False, False), (4, True, False), (4, True, True), (8, False, True)):
                values = torch.randn(batch, 130560, device='cuda')
                k = torch.full((batch,), 100, dtype=torch.int32, device='cuda') if topk else None
                p = torch.full((batch,), .95, device='cuda') if topp else None
                a, b = original(values.clone(), k, p), candidate(values.clone(), k, p)
                assert torch.equal(a, b)
                result['fallbacks'].append(dict(batch=batch, topk=topk, topp=topp, bitwise=True))
        frozen_check(result['source_snapshots'])
        assert len(result['masks']) == 156 and len(result['RNG_cases']) == 21 and len(result['events']) == 260
        result.update(phase='complete', complete=True, scope='Real API mask/RNG/graph gate; no model quality or service score')
    except BaseException as error:
        import traceback
        result.update(phase='failed', error=repr(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        save()
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
