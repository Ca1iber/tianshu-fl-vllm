# SPDX-License-Identifier: Apache-2.0
"""Add runtime-derived affine O64 W4 buffers to the unchanged r4 model."""
import atexit
import functools
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
EXP = BASE.parent

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def install(worker, mode, directory):
    import numpy as np
    import torch
    from torch._library.custom_ops import OPDEFS
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm_fl.worker.model_runner import ModelRunnerFL

    assert mode in ('control', 'candidate')
    assert worker.cache_config.kv_cache_memory_bytes is None
    assert worker.vllm_config.speculative_config is None
    assert worker.vllm_config.scheduler_config.max_num_batched_tokens == 4096
    assert worker.vllm_config.scheduler_config.max_num_seqs == 64
    assert not getattr(ModelRunnerFL, '_tianshu_o64_affine', False)
    ModelRunnerFL._tianshu_o64_affine = True
    gate = json.loads((BASE / 'CPU_GATE_32g.json').read_text())
    assert gate['complete']
    selection = json.loads((BASE / 'SELECTED_O_32g.json').read_text())
    expected = {b['prefix']: b for b in selection['banks']}
    assert len(expected) == 42
    component = EXP / 'tianshu_W4_attention64_20261024'
    quantizer = load('_o64_affine_verified_quantizer', component / 'common.py')
    scope = load('_o64_affine_scope', BASE / 'scope.py')
    definition = OPDEFS['tianshu_native_attention::linear']
    original_backend = definition._backend_fns['cuda']
    fn = definition._init_fn
    original_source = str(Path(inspect.getsourcefile(fn)).resolve())
    assert original_source == str(EXP / 'tianshu_native_int8_attention_service_20261022/policy.py')
    assert 'cuda' not in definition._disabled_kernel
    original_banks = dict(zip(fn.__code__.co_freevars, fn.__closure__))['banks'].cell_contents
    assert isinstance(original_banks, dict) and not original_banks
    banks = {}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(complete=False, mode=mode, phase='loading', banks=[], routes={},
        warmups=[], profile_probes=[], profile_windows=[],
        source=str(BASE / 'policy.py'), source_sha256=sha(BASE / 'policy.py'),
        scope_source=str(BASE / 'scope.py'), scope_sha256=sha(BASE / 'scope.py'),
        quantizer_source=str(component / 'common.py'), quantizer_sha256=sha(component / 'common.py'),
        selection_sha256=sha(BASE / 'SELECTED_O_32g.json'),
        original_op='tianshu_native_attention::linear', original_init_source=original_source,
        original_init_sha256=sha(original_source), original_backend_retained=True,
        public_register_kernel_used=True, original_BF16_W8_buffers_retained=True,
        no_checkpoint_or_model_structure_change=True, no_new_apply_override=True,
        scope='O only, FULL uniform64 decode, BF16 contiguous [64,2048], no LoRA/spec',
        additional_packed_payload_bytes=0, Python_calls_are_not_Graph_replays=True,
        equal_payload_in_control_and_candidate=True, no_global_tensor_workspace=True)

    def save():
        path = directory / f'route_{os.getpid()}.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(path)

    def thash(t):
        return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def memory_context(function):
        frame = inspect.currentframe()
        try:
            while frame:
                module = frame.f_globals.get('__name__')
                if frame.f_code.co_name == function:
                    if function == 'load_model' and module == 'vllm_fl.worker.model_runner' and 'm' in frame.f_locals:
                        profiler = frame.f_locals['m']
                        assert hasattr(profiler, 'initial_memory') and not hasattr(profiler, 'final_memory')
                        return dict(active=True, function=function, source=frame.f_code.co_filename,
                            source_sha256=sha(frame.f_code.co_filename), initial_memory=int(profiler.initial_memory))
                    if function == 'determine_available_memory' and module == 'vllm_fl.worker.worker':
                        return dict(active='profile_result' in frame.f_locals, function=function,
                            source=frame.f_code.co_filename, source_sha256=sha(frame.f_code.co_filename),
                            manual_KV_bytes=worker.cache_config.kv_cache_memory_bytes)
                frame = frame.f_back
            raise AssertionError('O64 packing/scratch outside actual native memory profiling')
        finally:
            del frame

    def compute(x, layer):
        return torch.ops.aten._weight_int4pack_mm_dotsaddz.default(
            x, layer._tianshu_o64_w, 128, layer._tianshu_o64_s,
            layer._tianshu_o64_z, size_n=2048)

    def zero_probes(context):
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state().clone()
        results = []
        for bid, layer in sorted(banks.items()):
            x = torch.zeros((64, 2048), device=layer.weight.device, dtype=torch.bfloat16)
            out = compute(x, layer)
            torch.cuda.synchronize()
            assert bool((out == 0).all())
            results.append(dict(bank_id=bid, M=64, zero_output=True, context=context))
            del x, out
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
        return results

    original_process = UnquantizedLinearMethod.process_weights_after_loading
    @functools.wraps(original_process)
    def process(method, layer):
        result = original_process(method, layer)
        prefix = getattr(layer, 'prefix', '')
        if prefix not in expected:
            return result
        proof = expected[prefix]
        context = memory_context('load_model')
        assert context['active'] and tuple(layer.weight.shape) == (2048, 2048)
        version = int(layer.weight._version)
        words = layer.weight.detach().view(torch.uint16).cpu().numpy().copy()
        assert quantizer.ahash(words) == proof['original_sha256']
        q, s, z, packed = quantizer.quant(words, 'affine')
        assert np.array_equal(quantizer.unpack(packed), q)
        entries = []
        for name, array, key in [('w', packed, 'packed_sha256'),
                                  ('s', s, 'scale_sha256'), ('z', z, 'zero_sha256')]:
            assert quantizer.ahash(array) == proof[key]
            tensor = torch.from_numpy(array.copy())
            if array.dtype == np.uint16:
                tensor = tensor.view(torch.bfloat16)
            tensor = tensor.to(layer.weight.device).contiguous()
            assert thash(tensor) == proof[key]
            layer.register_buffer('_tianshu_o64_' + name, tensor, persistent=False)
            entries.append(dict(name=name, shape=list(tensor.shape), stride=list(tensor.stride()),
                dtype=str(tensor.dtype), bytes=tensor.numel()*tensor.element_size(), sha256=proof[key]))
        bid = int(layer._tianshu_attention_bank)
        assert original_banks[bid]['layer'] is layer and original_banks[bid]['role'] == 'O'
        assert original_banks[bid]['prefix'] == prefix and bid not in banks
        assert int(layer.weight._version) == version and thash(layer.weight) == proof['original_sha256']
        banks[bid] = layer
        record['banks'].append(dict(bank_id=bid, prefix=prefix, layer=proof['layer'],
            original_sha256=proof['original_sha256'], entries=entries,
            original_weight_version_preserved=True, inside_DeviceMemoryProfiler=context))
        record['additional_packed_payload_bytes'] += sum(e['bytes'] for e in entries)
        if len(banks) == 42:
            record['warmups'] = zero_probes(memory_context('load_model'))
            record.update(load_complete=True, load_RNG_unchanged=True, phase='before_profile')
        save()
        return result
    UnquantizedLinearMethod.process_weights_after_loading = process

    original_profile = ModelRunnerFL.profile_run
    @functools.wraps(original_profile)
    def profile(owner, *args, **kwargs):
        previous = record['phase']
        record['phase'] = 'profile'
        context = memory_context('determine_available_memory')
        assert context['active'] and context['manual_KV_bytes'] is None
        window = dict(context=context, allocated_before=int(torch.cuda.memory_allocated()),
            peak_before=int(torch.cuda.max_memory_allocated()))
        try:
            result = original_profile(owner, *args, **kwargs)
            record['profile_probes'] = zero_probes(memory_context('determine_available_memory'))
            window['RNG_unchanged'] = True
            return result
        finally:
            torch.cuda.synchronize()
            window.update(allocated_after=int(torch.cuda.memory_allocated()),
                peak_after=int(torch.cuda.max_memory_allocated()))
            record['profile_windows'].append(window)
            record['phase'] = previous
            save()
    ModelRunnerFL.profile_run = profile

    original_capture = ModelRunnerFL.capture_model
    @functools.wraps(original_capture)
    def capture(owner, *args, **kwargs):
        record['phase'] = 'capture'
        save()
        try:
            result = original_capture(owner, *args, **kwargs)
            storages = {}
            for tensor in list(owner.model.parameters()) + list(owner.model.buffers()):
                if tensor.is_cuda:
                    storage = tensor.untyped_storage()
                    storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
            total = sum(storages.values())
            assert int(owner.model_memory_usage) + 16*1024**2 >= total
            assert len(banks) == len(record['warmups']) == len(record['profile_probes']) == 42
            assert record['additional_packed_payload_bytes'] == 93585408
            captured = [r for r in record['routes'].values() if r['phase'] == 'capture'
                        and r['capturing'] and r['implementation'] == 'native_affine_W4A16_O64']
            if mode == 'candidate':
                assert len({r['bank_id'] for r in captured}) == 42, 'O64 FULL graph scope not captured'
            record.update(complete=True, phase='runtime', capture_completed=True,
                model_memory_usage_bytes=int(owner.model_memory_usage),
                all_model_unique_parameter_buffer_storage_bytes=total,
                own_buffers_in_actual_model_accounting=True)
            return result
        finally:
            save()
    ModelRunnerFL.capture_model = capture

    def routed(x, weight, packed, scales, bank_id):
        actual = original_banks[bank_id]
        m = int(x.shape[0])
        if actual['role'] != 'O' or m != 64:
            return original_backend(x, weight, packed, scales, bank_id)
        try:
            context = get_forward_context()
            descriptor = context.batch_descriptor
            values = dict(mode=getattr(context.cudagraph_runtime_mode, 'name', str(context.cudagraph_runtime_mode)),
                num_tokens=getattr(descriptor, 'num_tokens', None),
                num_reqs=getattr(descriptor, 'num_reqs', None),
                uniform=getattr(descriptor, 'uniform', False), has_lora=getattr(descriptor, 'has_lora', False))
        except AssertionError:
            values = dict(mode='MISSING', num_tokens=None, num_reqs=None, uniform=False, has_lora=False)
        eligible = scope.eligible(actual['role'], x.shape, x.dtype == torch.bfloat16,
            x.is_contiguous(), values['mode'], values['num_tokens'], values['num_reqs'],
            values['uniform'], values['has_lora'])
        use = mode == 'candidate' and eligible
        capturing = torch.cuda.is_current_stream_capturing()
        key = str((bank_id, record['phase'], capturing, tuple(values.items()), use))
        row = record['routes'].setdefault(key, dict(bank_id=bank_id, prefix=actual['prefix'], M=m,
            phase=record['phase'], capturing=capturing, context=values, eligible=eligible,
            implementation='native_affine_W4A16_O64' if use else 'unchanged_r4_backend', calls=0))
        row['calls'] += 1
        if use:
            layer = banks[bank_id]
            assert actual['layer'] is layer and weight.data_ptr() == layer.weight.data_ptr()
            assert packed.data_ptr() == layer._tianshu_attention_packed.data_ptr()
            assert scales.data_ptr() == layer._tianshu_attention_scales.data_ptr()
            out = compute(x, layer)
        else:
            out = original_backend(x, weight, packed, scales, bank_id)
        if bank_id == 83 and row['calls'] & (row['calls']-1) == 0:
            save()
        return out
    definition.register_kernel('cuda', routed)
    record['CUDA_route_registered'] = True
    atexit.register(save)
    save()
