# SPDX-License-Identifier: Apache-2.0
"""Exercise pinned native dispatcher plus generated dummy metadata on the CPU."""
import argparse
import ast
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import textwrap
from types import SimpleNamespace as NS
from ragged_policy import GRAPH_PLANS, capture_queries, install, plan, render_dummy


class Mode(Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2
    FULL_DECODE_ONLY = 3

    def mixed_mode(self):
        return self.NONE

    def decode_mode(self):
        return self.FULL

    def separate_routine(self):
        return self == self.FULL_DECODE_ONLY

    def has_mode(self, mode):
        return mode == self.FULL and self in (self.FULL, self.FULL_DECODE_ONLY)

    def is_valid_runtime_mode(self):
        return self != self.FULL_DECODE_ONLY

    @classmethod
    def valid_runtime_modes(cls):
        return {cls.NONE, cls.FULL, cls.PIECEWISE}


def extract_class(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[name]


def config():
    hf = NS(hidden_size=2048, intermediate_size=6144, num_hidden_layers=42,
            num_attention_heads=16, num_key_value_heads=2)
    model = NS(hf_text_config=hf, max_model_len=131072, quantization=None,
               is_multimodal_model=False, is_encoder_decoder=False, multimodal_config=None)
    compilation = NS(cudagraph_mode=Mode.FULL_DECODE_ONLY, max_cudagraph_capture_size=128,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128], compile_sizes=[])
    return NS(model_config=model, speculative_config=NS(method='ngram', num_speculative_tokens=1,
                  prompt_lookup_min=3, prompt_lookup_max=5), lora_config=None,
              parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, use_ubatching=False),
              scheduler_config=NS(max_num_batched_tokens=2048, max_num_seqs=256),
              compilation_config=compilation, observability_config=NS(cudagraph_metrics=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dispatcher', type=Path, required=True)
    parser.add_argument('--forward-context', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    args = parser.parse_args()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(args.dispatcher) == '51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63'
    assert sha(args.forward_context) == '1c6fa51e9a02a120fed5e7357afb1190516cbdfefd1c644dfcd860603f1527c5'
    ns = dict(dataclass=dataclass)
    descriptor = extract_class(args.forward_context, 'BatchDescriptor', ns)
    ns = dict(replace=replace, product=product, CUDAGraphMode=Mode, VllmConfig=object,
              BatchDescriptor=descriptor, AbstractSet=set, logger=NS(debug=lambda *a: None))
    dispatcher_cls = extract_class(args.dispatcher, 'CudagraphDispatcher', ns)
    cfg = config()
    dispatcher = object.__new__(dispatcher_cls)
    dispatcher.vllm_config = cfg
    dispatcher.compilation_config = cfg.compilation_config
    dispatcher.uniform_decode_query_len = 2
    dispatcher.cudagraph_keys = {Mode.FULL: set(), Mode.PIECEWISE: set()}
    dispatcher.keys_initialized = False
    dispatcher.specialize_lora_count = False
    dispatcher.cudagraph_mode = Mode.NONE

    class Runner:
        vllm_config = cfg
        cudagraph_dispatcher = dispatcher
        scheduler_config = cfg.scheduler_config
        max_num_tokens = 2048
        uniform_decode_query_len = 2
        input_batch = NS(num_computed_tokens_cpu=[4096]*64, num_prompt_tokens=[4096]*64)
        model = NS(concrete_graph_entries={})
        last_base = None

        def _determine_batch_execution_and_padding(self, num_tokens, num_reqs, num_scheduled_tokens_np,
                max_num_scheduled_tokens, use_cascade_attn, allow_microbatching=True, force_eager=False,
                force_uniform_decode=None, force_has_lora=None, force_num_active_loras=None, num_encoder_reqs=0):
            uniform = (max_num_scheduled_tokens == 2 and num_tokens == 2*num_reqs) if force_uniform_decode is None else force_uniform_decode
            mode, desc = self.cudagraph_dispatcher.dispatch(num_tokens,
                uniform_decode=uniform, has_lora=False,
                valid_modes={Mode.NONE} if force_eager else None,
                invalid_modes={Mode.FULL} if use_cascade_attn else None)
            self.last_base = mode, desc, False, None, None
            return self.last_base

        def _dummy_run(self, *args, **kwargs):
            return 'unchanged native dummy'

        def _warmup_and_capture(self, desc, cudagraph_runtime_mode, fail=False):
            if fail:
                raise ValueError('original warm exception')
            q = capture_queries(desc.num_reqs, desc.num_tokens)
            for eager in (True, False):
                result = self._determine_batch_execution_and_padding(desc.num_tokens, desc.num_reqs, q,
                    2, False, force_eager=eager, force_uniform_decode=False)
                assert result[0] == (Mode.NONE if eager else Mode.FULL)
                if not eager:
                    assert result[1] == desc
            self.model.concrete_graph_entries[desc] = NS(graph=object())
            return 'native warm return'

    identity_decorator = lambda: lambda fn: fn
    module = NS(__file__=str(args.runner), ModelRunnerFL=Runner, CudagraphDispatcher=dispatcher_cls,
                managed_inference_mode=identity_decorator, CUDAGraphMode=Mode)
    generated, derivation = render_dummy(args.runner.read_bytes())
    # Execute the actual generated dummy metadata prefix, stopping before GPU work.
    tree = ast.parse(generated)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    stop = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'num_scheduled_tokens' for t in n.targets))
    fn.body = fn.body[:stop] + [ast.Return(value=ast.Tuple(elts=[ast.Name(id=x, ctx=ast.Load())
        for x in ('max_query_len', 'num_reqs', 'num_scheduled_tokens_list')], ctx=ast.Load()))]
    fn.returns = None
    fn.decorator_list = []
    for a in fn.args.args:
        a.annotation = None
    metadata_ns = dict(CUDAGraphMode=Mode)
    exec(compile(ast.fix_missing_locations(tree), '<actual-generated-metadata>', 'exec'), metadata_ns)
    with TemporaryDirectory() as tmp:
        owner = Runner()
        state = install(module, Mode, descriptor, tmp)
        dispatcher.initialize_cudagraph_keys(Mode.FULL_DECODE_ONLY, 2)
        expected = {descriptor(t, n, False) for n, t in GRAPH_PLANS}
        assert not expected.intersection(dispatcher.cudagraph_keys[Mode.FULL])
        assert all(k.uniform for k in dispatcher.cudagraph_keys[Mode.FULL])
        native_keys = set(dispatcher.cudagraph_keys[Mode.FULL])
        groups = dispatcher.get_capture_descs()
        assert len(groups) == 1 and groups[0][0] == Mode.FULL
        assert set(groups[0][1]) == expected | native_keys
        assert len(expected) == 248
        metadata_cases = 0
        for n, t in GRAPH_PLANS:
            owner._tianshu_ragged_capture = n, t
            assert metadata_ns['_dummy_run'](owner, t, Mode.NONE) == (2, n, capture_queries(n, t))
            owner._tianshu_ragged_capture = None
            key = descriptor(t, n, False)
            assert owner._warmup_and_capture(key, Mode.FULL) == 'native warm return'
            assert owner._tianshu_ragged_capture is None
            metadata_cases += 1
        assert state['dummy_warm_calls'] == state['dummy_capture_calls'] == 248
        assert owner._dummy_run(17) == 'unchanged native dummy'
        try:
            owner._warmup_and_capture(descriptor(80, 64, False), Mode.FULL, fail=True)
        except ValueError:
            assert owner._tianshu_ragged_capture is None
        else:
            raise AssertionError('Original exception swallowed')
        eligible = 0
        fallbacks = 0
        for n in range(1, 65):
            for twos in range(n+1):
                q = [2]*twos + [1]*(n-twos)
                out = owner._determine_batch_execution_and_padding(sum(q), n, q, max(q), False)
                chosen = plan(sum(q), n, max(q), q)
                if chosen is None or owner.last_base[0] != Mode.NONE:
                    assert out is owner.last_base
                    assert owner.last_base[0] == (Mode.FULL if twos == n else Mode.NONE)
                    fallbacks += 1
                else:
                    pn, pt = chosen
                    assert out == (Mode.FULL, descriptor(pt, pn, False), False, None, None)
                    assert pt >= sum(q) and pn == n and pt*8 <= sum(q)*9
                    eligible += 1
        q = [2]*15 + [1]*49
        for options in (dict(force_eager=True), dict(force_uniform_decode=False), dict(use_cascade_attn=True),
                        dict(num_encoder_reqs=1)):
            kwargs = dict(num_tokens=79, num_reqs=64, num_scheduled_tokens_np=q,
                          max_num_scheduled_tokens=2, use_cascade_attn=False)
            kwargs.update(options)
            assert owner._determine_batch_execution_and_padding(**kwargs) is owner.last_base
        owner.input_batch.num_computed_tokens_cpu[0] = 0
        assert owner._determine_batch_execution_and_padding(79, 64, q, 2, False) is owner.last_base
        owner.input_batch.num_computed_tokens_cpu[0] = 4096
        key = descriptor(80, 64, False)
        owner.model.concrete_graph_entries[key].graph = None
        assert owner._determine_batch_execution_and_padding(79, 64, q, 2, False) is owner.last_base
        # Short prefill and all-Q1 cannot auto-match the owned (tokens==requests) keys.
        q = [3]*8 + [2]*8 + [1]*16
        assert sum(q) == 56
        assert owner._determine_batch_execution_and_padding(56, 32, q, 3, False)[0] == Mode.NONE
        assert dispatcher.dispatch(64, uniform_decode=False)[0] == Mode.NONE
        assert dispatcher.cudagraph_keys[Mode.FULL] == native_keys
    print(json.dumps(dict(complete=True, native_dispatcher_and_descriptor_executed=True,
        generated_metadata_cases=metadata_cases, mixed_routes=eligible, native_fallback_cases=fallbacks,
        boundary_fallbacks=6, original_warm_exception_and_return_preserved=True,
        derivation=derivation, GPU_test=False)))


if __name__ == '__main__':
    main()
