# SPDX-License-Identifier: Apache-2.0
"""Isolated bounded mixed-Q1/Q2 graph keys, metadata capture and bounded routing."""
import ast
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import textwrap

RUNNER_SHA = 'b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'
GRAPH_PLANS = tuple((n, t) for n in range(16, 65) for t in range(((n+8)//8)*8, 2*n+1, 8))


def plan(tokens, requests, max_query, query_lengths):
    values = tuple(int(v) for v in query_lengths)
    if (not 16 <= requests <= 64 or len(values) != requests or max_query != 2
            or not all(v in (1, 2) for v in values) or min(values) != 1
            or max(values) != 2 or sum(values) != tokens):
        return None
    padded_requests = requests  # Keep request count and its attention tile unchanged
    padded_tokens = max(padded_requests, ((int(tokens) + 7) // 8) * 8)
    selected = padded_requests, padded_tokens
    return selected if selected in GRAPH_PLANS and padded_tokens*8 <= tokens*9 else None


def capture_queries(requests, capacity):
    assert (requests, capacity) in GRAPH_PLANS
    twos = max(1, capacity - requests)
    zeros = max(0, requests + twos - capacity)
    values = [2]*twos + [1]*(requests-twos-zeros) + [0]*zeros
    assert len(values) == requests and sum(values) == capacity and max(values) == 2
    return values


def render_dummy(raw):
    """Change only dummy metadata assignments; retain model/attention implementation."""
    assert hashlib.sha256(raw).hexdigest() == RUNNER_SHA
    tree = ast.parse(raw.decode())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelRunnerFL')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_dummy_run')
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    original = textwrap.dedent('\n'.join(raw.decode().splitlines()[start - 1:node.end_lineno]) + '\n')
    old_max = 'max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens'
    new_max = 'max_query_len = 2  # Dedicated mixed-Q1/Q2 capture, not uniform decode'
    anchor = '    assert sum(num_scheduled_tokens_list) == num_tokens\n'
    insertion = ('    num_reqs, expected_tokens = self._tianshu_ragged_capture\n'
                 '    assert expected_tokens == num_tokens and not uniform_decode\n'
                 '    twos = max(1, num_tokens - num_reqs)\n'
                 '    zeros = max(0, num_reqs + twos - num_tokens)\n'
                 '    num_scheduled_tokens_list = ([2]*twos + [1]*(num_reqs-twos-zeros)\n'
                 '                                 + [0]*zeros)\n')
    assert original.count(old_max) == original.count(anchor) == 1
    generated = original.replace(old_max, new_max).replace(anchor, insertion + anchor)
    assert generated.replace(new_max, old_max).replace(insertion, '') == original
    ast.parse(generated)
    return generated, dict(original_runner_sha256=RUNNER_SHA,
        original_dummy_sha256=hashlib.sha256(original.encode()).hexdigest(),
        generated_dummy_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        only_dummy_query_length_and_query_list_metadata_changed=True,
        model_and_attention_kernel_bodies_changed=False, installed_sources_modified=False)


def supported_config(config):
    spec = config.speculative_config
    hf = config.model_config.hf_text_config
    parallel = config.parallel_config
    return bool(spec is not None and spec.method == 'ngram' and spec.num_speculative_tokens == 1
        and (spec.prompt_lookup_min, spec.prompt_lookup_max) == (3, 5)
        and config.model_config.max_model_len == 131072
        and tuple(getattr(hf, k, None) for k in ('hidden_size', 'intermediate_size', 'num_hidden_layers',
            'num_attention_heads', 'num_key_value_heads')) == (2048, 6144, 42, 16, 2)
        and config.model_config.quantization is None and not config.model_config.is_multimodal_model
        and not config.model_config.is_encoder_decoder and config.lora_config is None
        and parallel.tensor_parallel_size == parallel.pipeline_parallel_size == parallel.data_parallel_size == 1
        and not parallel.use_ubatching and config.scheduler_config.max_num_batched_tokens == 2048
        and config.scheduler_config.max_num_seqs == 256
        and config.compilation_config.cudagraph_mode.name == 'FULL_DECODE_ONLY'
        and not config.observability_config.cudagraph_metrics)


def install(module, modes, descriptor, directory, *, record=None):
    directory = Path(directory)
    source, derivation = render_dummy(Path(module.__file__).read_bytes())
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / 'generated_ragged_dummy.py'
    if output.exists():
        assert output.read_text() == source
    else:
        output.write_text(source)
    namespace = dict(vars(module))
    exec(compile('from __future__ import annotations\n' + source, str(output), 'exec'), namespace)
    mixed_dummy = namespace['_dummy_run']
    cls = module.ModelRunnerFL
    dispatcher_class = module.CudagraphDispatcher
    original_init = dispatcher_class.initialize_cudagraph_keys
    original_capture_descs = dispatcher_class.get_capture_descs
    original_dummy = cls._dummy_run
    original_warm = cls._warmup_and_capture
    original_dispatch = cls._determine_batch_execution_and_padding
    signature = inspect.signature(original_dispatch)
    state = dict(derivation=derivation, generated_source=str(output),
                 dummy_warm_calls=0, dummy_capture_calls=0, runtime_calls=0,
                 real_rows=0, padded_rows=0, runtime_plans={}, registered_keys=[])

    def save():
        if record:
            record(state)

    def mixed_key(requests, capacity):
        return descriptor(num_tokens=capacity, num_reqs=requests, uniform=False,
                          has_lora=False, num_active_loras=0)

    @wraps(original_init)
    def initialize(owner, cudagraph_mode, uniform_decode_query_len=1):
        result = original_init(owner, cudagraph_mode, uniform_decode_query_len)
        if supported_config(owner.vllm_config):
            assert cudagraph_mode.name == 'FULL_DECODE_ONLY' and owner.uniform_decode_query_len == 2
            owner._tianshu_mixed_graph_keys = {mixed_key(n, t) for n, t in GRAPH_PLANS}
            assert not owner._tianshu_mixed_graph_keys.intersection(owner.cudagraph_keys[modes.FULL])
            state['registered_keys'] = [dict(tokens=t, requests=n, uniform=False) for n, t in GRAPH_PLANS]
            state['native_dispatcher_key_set_unchanged'] = True
            save()
        return result

    @wraps(original_capture_descs)
    def capture_descs(owner):
        groups = original_capture_descs(owner)
        keys = getattr(owner, '_tianshu_mixed_graph_keys', set())
        if not keys or not owner.keys_initialized or not supported_config(owner.vllm_config):
            return groups
        # Keep one group per runtime mode for native graph-memory estimation.
        # Owned mixed keys are not exposed to native dispatch: N==tokens keys
        # could otherwise accidentally match all-Q1 or short-prefill batches.
        result = []
        found = False
        for mode, descs in groups:
            if mode == modes.FULL:
                descs = sorted(set(descs) | keys,
                               key=lambda d: (d.num_tokens, d.num_reqs or 0, d.uniform), reverse=True)
                found = True
            result.append((mode, descs))
        assert found
        return result

    @wraps(original_warm)
    def warm(owner, desc, cudagraph_runtime_mode, *args, **kwargs):
        eligible = (supported_config(owner.vllm_config) and cudagraph_runtime_mode == modes.FULL
                    and desc in {mixed_key(n, t) for n, t in GRAPH_PLANS})
        if not eligible:
            return original_warm(owner, desc, cudagraph_runtime_mode, *args, **kwargs)
        assert getattr(owner, '_tianshu_ragged_capture', None) is None
        owner._tianshu_ragged_capture = desc.num_reqs, desc.num_tokens
        try:
            return original_warm(owner, desc, cudagraph_runtime_mode, *args, **kwargs)
        finally:
            owner._tianshu_ragged_capture = None

    @wraps(original_dummy)
    def dummy(owner, *args, **kwargs):
        if getattr(owner, '_tianshu_ragged_capture', None) is None:
            return original_dummy(owner, *args, **kwargs)
        assert not kwargs.get('uniform_decode', False)
        return mixed_dummy(owner, *args, **kwargs)

    @wraps(original_dispatch)
    def dispatch(owner, *args, **kwargs):
        original = original_dispatch(owner, *args, **kwargs)
        if not supported_config(owner.vllm_config):
            return original
        bound = signature.bind(owner, *args, **kwargs).arguments
        capture = getattr(owner, '_tianshu_ragged_capture', None)
        if capture is not None:
            requests, capacity = capture
            assert bound['num_tokens'] == capacity and bound['num_reqs'] == requests
            assert bound['max_num_scheduled_tokens'] == 2 and bound.get('force_uniform_decode') is False
            assert list(bound['num_scheduled_tokens_np']) == capture_queries(requests, capacity)
            if bound.get('force_eager', False):
                assert original[0] == modes.NONE
                state['dummy_warm_calls'] += 1
                save()
                return original
            assert original[2:] == (False, None, None)
            assert mixed_key(requests, capacity) in owner.cudagraph_dispatcher._tianshu_mixed_graph_keys
            state['dummy_capture_calls'] += 1
            save()
            return modes.FULL, mixed_key(requests, capacity), False, None, None
        if (original[0] != modes.NONE or original[2:] != (False, None, None)
                or bound.get('force_eager', False) or bound.get('force_uniform_decode') is not None
                or bound['use_cascade_attn'] or bound.get('num_encoder_reqs', 0)):
            return original
        selected = plan(bound['num_tokens'], bound['num_reqs'], bound['max_num_scheduled_tokens'],
                        bound['num_scheduled_tokens_np'])
        if selected is None:
            return original
        requests, capacity = selected
        actual_requests = bound['num_reqs']
        batch = owner.input_batch
        if any(a < b for a, b in zip(batch.num_computed_tokens_cpu[:actual_requests], batch.num_prompt_tokens[:actual_requests])):
            return original
        key = mixed_key(requests, capacity)
        entry = getattr(owner.model, 'concrete_graph_entries', {}).get(key)
        if key not in owner.cudagraph_dispatcher._tianshu_mixed_graph_keys or entry is None or entry.graph is None:
            return original
        state['runtime_calls'] += 1
        state['real_rows'] += int(bound['num_tokens'])
        state['padded_rows'] += capacity
        name = f'{requests}x{capacity}'
        first = name not in state['runtime_plans']
        state['runtime_plans'][name] = state['runtime_plans'].get(name, 0) + 1
        if first or state['runtime_calls'] % 128 == 0:
            save()
        return modes.FULL, key, False, None, None

    dispatcher_class.initialize_cudagraph_keys = initialize
    dispatcher_class.get_capture_descs = capture_descs
    cls._warmup_and_capture = warm
    cls._dummy_run = dummy
    cls._determine_batch_execution_and_padding = dispatch
    return state
