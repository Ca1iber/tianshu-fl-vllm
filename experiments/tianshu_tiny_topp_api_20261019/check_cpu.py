# SPDX-License-Identifier: Apache-2.0
"""Check routing boundaries and the inspected sampler's forwarding contract."""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import importlib.abc
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

from api_route import SAMPLER_SHA, build_route


@dataclass(frozen=True)
class Device:
    type: str = 'cuda'
    index: int = 0


class Tensor:
    def __init__(self, shape, dtype='f32', device=Device(), contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim = len(shape)
        self.contiguous = contiguous
        self.probs = object()
        self.processed = object()

    def is_contiguous(self):
        return self.contiguous

    def softmax(self, dim, dtype):
        assert dim == -1 and dtype == 'f32'
        return self.probs

    def log_softmax(self, dim, dtype):
        assert dim == -1 and dtype == 'f32'
        return self.processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.sampler_source.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SAMPLER_SHA
    calls = []

    def native(logits, k, p):
        calls.append(('native', logits, k, p))
        return logits

    def candidate(logits, k, p):
        calls.append(('candidate', logits, k, p))
        return logits

    torch = SimpleNamespace(float32='f32', Tensor=Tensor, Generator=object)
    route = build_route(native, SimpleNamespace(apply_top_k_top_p_triton=candidate), torch)
    count = 0

    def check(logits, k, p, expected):
        nonlocal count
        before = len(calls)
        assert route(logits, k, p) is logits
        assert len(calls) == before + 1
        actual, x, actual_k, actual_p = calls[-1]
        assert actual == expected and x is logits and actual_k is k and actual_p is p
        count += 1

    for batch in range(0, 10):
        check(Tensor((batch, 130560)), None, Tensor((batch,)),
              'candidate' if 1 <= batch <= 7 else 'native')
    x, p = Tensor((4, 130560)), Tensor((4,))
    check(x, None, None, 'native')
    check(x, Tensor((4,), dtype='i32'), p, 'native')
    for altered in (Tensor((4,)), Tensor((4, 130559)), Tensor((4, 130560), dtype='bf16'),
                    Tensor((4, 130560), device=Device('cpu')),
                    Tensor((4, 130560), contiguous=False)):
        check(altered, None, p, 'native')
    for altered in (Tensor((4, 1)), Tensor((3,)), Tensor((4,), dtype='bf16'),
                    Tensor((4,), device=Device('cpu')), Tensor((4,), device=Device(index=1)),
                    Tensor((4,), contiguous=False)):
        check(x, None, altered, 'native')

    # Execute the actual pinned forward_native body with instrumented tensors.
    # This verifies masks, optional returned logits and RNG arguments together;
    # it does not claim actual GPU RNG numerical equivalence.
    tree = ast.parse(source.decode())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TopKTopPSampler')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_native')
    body = textwrap.dedent('\n'.join(source.decode().splitlines()[node.lineno-1:node.end_lineno]))
    random_calls = []
    sampled = object()

    def random_sample(probs, generators, fp64):
        random_calls.append((probs, generators, fp64))
        return sampled

    namespace = dict(torch=torch, apply_top_k_top_p=route, random_sample=random_sample)
    exec(compile(body, str(args.sampler_source), 'exec'), namespace)
    forwarding = 0
    for batch in (1, 2, 4, 7, 8):
        for mode in ('raw_logprobs', 'processed_logits', 'processed_logprobs'):
            for fp64 in (False, True):
                logits, probabilities = Tensor((batch, 130560)), Tensor((batch,))
                generators = {0: object()}
                before = dict(generators)
                result, extra = namespace['forward_native'](
                    SimpleNamespace(logprobs_mode=mode, use_fp64_gumbel=fp64),
                    logits, generators, None, probabilities)
                assert result is sampled
                expected_extra = logits if mode == 'processed_logits' else logits.processed if mode == 'processed_logprobs' else None
                assert extra is expected_extra
                probs, observed_generators, observed_fp64 = random_calls[-1]
                assert probs is logits.probs and observed_generators is generators
                assert observed_fp64 is fp64 and generators == before
                assert calls[-1][0] == ('candidate' if batch <= 7 else 'native')
                forwarding += 1
    assert len(random_calls) == forwarding == 30
    hook_path = Path(__file__).with_name('hooks') / 'sitecustomize.py'
    hook_source = hook_path.read_text()
    finder_node = next(n for n in ast.parse(hook_source).body
                       if isinstance(n, ast.ClassDef) and n.name == 'Finder')
    finder_body = '\n'.join(hook_source.splitlines()[finder_node.lineno-1:finder_node.end_lineno])
    finder_namespace = dict(importlib=__import__('importlib'), sys=sys,
                            Loader=lambda loader: SimpleNamespace(original=loader))
    exec(compile(finder_body, str(hook_path), 'exec'), finder_namespace)
    cls = finder_namespace['Finder']
    original_chain = sys.meta_path
    hook_checks = 0
    try:
        for outcome in ('success', 'missing', 'raises'):
            finder = cls()
            seen = []
            loader = object()
            spec = SimpleNamespace(loader=loader)

            def skip(*args):
                seen.append('skip')
                return None

            def resolve(*args):
                seen.append('resolve')
                if outcome == 'raises':
                    raise ValueError('sentinel-loader-error')
                return spec if outcome == 'success' else None

            chain = [finder, SimpleNamespace(find_spec=skip), SimpleNamespace(find_spec=resolve)]
            before_chain = list(chain)
            sys.meta_path = chain
            try:
                actual = finder.find_spec('vllm.v1.sample.ops.topk_topp_sampler', ['sentinel-path'])
            except (ValueError, AssertionError) as error:
                assert outcome != 'success'
                assert isinstance(error, ValueError if outcome == 'raises' else AssertionError)
            else:
                assert outcome == 'success' and actual is spec
                assert actual.loader.original is loader
            assert seen == ['skip', 'resolve']
            assert len(sys.meta_path) == len(before_chain)
            assert all(a is b for a, b in zip(sys.meta_path, before_chain))
            assert finder.find_spec('other.module') is None
            assert seen == ['skip', 'resolve']
            hook_checks += 1
    finally:
        sys.meta_path = original_chain
    result = dict(complete=True, routing_boundaries=count, actual_forward_native_contract_cases=forwarding,
                  import_delegation_and_restoration_cases=hook_checks,
                  sampler_source_sha256=SAMPLER_SHA,
                  api_route_sha256=hashlib.sha256(Path(__file__).with_name('api_route.py').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  hook_sha256=hashlib.sha256(hook_path.read_bytes()).hexdigest(),
                  scope='CPU stubs and actual pinned method, no GPU numerical or RNG-quality claim',
                  GPU_run=False, model_run=False, adopted=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
