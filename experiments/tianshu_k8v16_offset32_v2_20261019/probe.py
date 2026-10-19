# SPDX-License-Identifier: Apache-2.0
"""Compare actual K8V16 offset64/offset32 with unchanged values and launch policy."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import runpy
import signal
import statistics
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stop-at-utc', required=True)
    a = p.parse_args()
    root = a.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    remaining = int((datetime.fromisoformat(a.stop_at_utc) - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 1200
    signal.alarm(min(remaining, 1800))
    hardware = subprocess.check_output(['ixsmi'], text=True)
    assert 'No running processes found' in hardware
    (root / 'hardware_before.txt').write_text(hardware)
    base = Path(__file__).resolve().parent
    sys.path.insert(0, str(base.parent / 'tianshu_gemm_topp_geo_20261016'))
    from validation_profile_new import prepare
    command, environment, profile, sources = prepare(root, Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
    os.environ.clear()
    os.environ.update(environment)
    runpy.run_path(str(Path(profile['PYTHONPATH']) / 'sitecustomize.py'))
    import torch
    import vllm_fl
    import vllm.v1.attention.backends.triton_attn as backend
    from vllm.v1.attention.ops import triton_unified_attention as native
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
    from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
    from vllm_fl.worker import iluvatar_adaptive_decode as adaptive
    from vllm_fl.worker import iluvatar_hybrid_attention as hybrid
    from types import SimpleNamespace
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=2048,
        num_hidden_layers=42, num_attention_heads=16, num_key_value_heads=2), dtype=torch.bfloat16,
        quantization=None, is_multimodal_model=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1,
            data_parallel_size=1, decode_context_parallel_size=1), lora_config=None, speculative_config=None)
    assert adaptive.configure(config, logging.getLogger('mixed_planes'))
    assert hybrid.configure(config, logging.getLogger('mixed_planes'))
    import writer
    from layout import MODES, allocate
    from plans import configuration, PLANS
    sys.path.insert(0, str(base.parent / 'tianshu_soft8_engine_v2_20261019'))
    from dispatch_policy import choose_config
    spec = importlib.util.spec_from_file_location('_mixed_planes_parent', base.parent / 'tianshu_soft8_engine_v2_20261019/source.py')
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    cv = Path(sys.modules['tianshu_native_cache_policy'].__file__)
    import inspect
    original_writer = Path(inspect.getsourcefile(backend.triton_reshape_and_cache_flash))
    reader, unused_writer, proof = parent.load_modules(cv, original_writer, root / 'codec_generated')
    variant_spec = importlib.util.spec_from_file_location('_k8v16_offset32_exact_owner', base / 'source.py')
    variant_module = importlib.util.module_from_spec(variant_spec)
    variant_spec.loader.exec_module(variant_module)
    assert Path(variant_module.__file__).resolve() == base / 'source.py'
    assert_safe = variant_module.assert_safe
    reader32, address32_proof = variant_module.load_variant(reader, root / 'address32_generated')
    for gate_name in ('CPU_GATE_32g.json', 'CPU_PRODUCER_GATE_32g.json'):
        gate = json.loads((base / gate_name).read_text())
        assert gate['complete']
        for name, sha in gate['sources'].items():
            assert hashlib.sha256((base / name).read_bytes()).hexdigest() == sha
    prerequisite = json.loads((base / 'PREREQUISITE_32g.json').read_text())
    qualified = Path(prerequisite['remote_result'])
    assert hashlib.sha256(qualified.read_bytes()).hexdigest() == prerequisite['remote_result_sha256']
    assert json.loads(qualified.read_text())['phase'] == 'complete'
    for path, sha in prerequisite['frozen_sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
        assert path not in sources or sources[path] == sha
        sources[path] = sha
    paths = [base / n for n in ('probe.py', 'writer.py', 'layout.py', 'plans.py', 'check_cpu.py', 'run_probe.py',
        'CPU_GATE_32g.json', 'CPU_PRODUCER_GATE_32g.json', 'PREREQUISITE_32g.json', 'source.py', 'check_producer_cpu.py')]
    paths.append(writer.OWNER)
    paths += [Path(parent.__file__), Path(reader.__file__), Path(reader32.__file__), Path(unused_writer.__file__),
        Path(inspect.getsourcefile(_reshape_attention_kv_cache)), Path(inspect.getsourcefile(FullAttentionSpec)),
        base.parent / 'tianshu_soft8_engine_v2_20261019/dispatch_policy.py']
    sources.update({str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
    for path, sha in sources.items():
        dest = root / 'source_snapshot' / Path(path).relative_to('/')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(path).read_bytes())
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == sha
    data = dict(phase='codec', complete=False, official_benchmark=False, adopted=False,
        source_snapshots=sources, profile_environment=profile, server_command=command,
        codec_proof=proof, address32_proof=address32_proof, offset_bounds={}, plans=PLANS, modes=list(MODES), cases=[], producer_records={},
        scope='Actual native padded-page fixtures and writer+decode; no model runner, full engine, prefill or Math claim')

    def save():
        tmp = root / 'result.tmp'
        tmp.write_text(json.dumps(data, indent=2) + '\n')
        tmp.replace(root / 'result.json')

    save()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False

    def encode(x):
        bits = x.view(torch.int16).to(torch.int32)
        mag = bits & 32767
        rounded = ((mag + 7 + ((mag >> 4) & 1)) >> 4) - 928
        return (torch.where(rounded >= 8, rounded.clamp(max=127), 0) | ((bits & 32768) >> 8)).to(torch.uint8)

    def decode(x):
        if x.dtype != torch.uint8:
            return x
        bits = x.to(torch.int32)
        mag = bits & 127
        return (torch.where(mag >= 8, (mag << 4) + 14848, 0) | ((bits & 128) << 8)).to(torch.int16).view(torch.bfloat16)

    def check(x, ref):
        diff = (x.float() - ref).abs()
        bad = diff > .02 + .01 * ref.abs()
        return dict(passed=bool(torch.isfinite(x).all() and torch.isfinite(ref).all() and not bad.any()),
            failed_elements=int(bad.sum()), max_abs=float(diff.max()), atol=.02, rtol=.01)

    class Producer:
        def __init__(self, kernel):
            self.kernel, self.mode, self.tile = kernel, None, None

        def __getitem__(self, grid):
            def launch(*args, **kw):
                launch_grid = grid
                assert not args and kw['KV_QUANT_MODE'] == 0 and kw['HEAD_SIZE'] == 128
                n = kw['num_seqs']
                assert kw['IS_3D'] and n <= 64
                plan = configuration(self.mode, n, current_context[0])
                assert kw['NUM_SEGMENTS_PER_SEQ'] == plan['segments']
                kw.update(TILE_SIZE=plan['TILE_SIZE'], num_warps=plan['num_warps'], num_stages=plan['num_stages'])
                self.tile = kw['TILE_SIZE']
                kw['KV_CACHE_MODIFIER'] = '.cv'
                result = self.kernel[self.mode][launch_grid](**kw)
                key = str((self.mode, n, current_context[0]))
                data['producer_records'][key] = dict(mode=self.mode, batch=n, context=current_context[0],
                    key_dtype=str(kw['key_cache_ptr'].dtype), value_dtype=str(kw['value_cache_ptr'].dtype),
                    IS_3D=kw['IS_3D'], tile=self.tile, warps=result.metadata.num_warps,
                    stages=result.metadata.num_stages, segments=kw['NUM_SEGMENTS_PER_SEQ'],
                    n_regs=getattr(result, 'n_regs', None), n_spills=getattr(result, 'n_spills', None))
                return result
            return launch

    producer = Producer(dict(current=reader.kernel_unified_attention, address32=reader32.kernel_unified_attention))
    raw_reduce = dict(current=reader.reduce_segments, address32=reader32.reduce_segments)
    scales = torch.ones(1, device='cuda')

    class Reducer:
        def __getitem__(self, grid):
            def launch(*args, **kw):
                kw['TILE_SIZE'] = producer.tile
                return raw_reduce[producer.mode][grid](*args, **kw)
            return launch

    reader.kernel_unified_attention, reader.reduce_segments = producer, Reducer()
    current_context = [0]
    try:
        bits = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
        encoded = torch.empty(65536, device='cuda', dtype=torch.uint8)
        writer.encode_all[(256,)](bits.view(torch.bfloat16), encoded, 65536, num_warps=1, num_stages=1)
        data['all65536_actual_writer_encoded'] = encoded.cpu().tolist()
        assert data['all65536_actual_writer_encoded'] == [parent.encode_bits(i) for i in range(65536)]
        decoded = torch.empty(256, device='cuda', dtype=torch.bfloat16)
        reader._codec_all_codes[(1,)](torch.arange(256, device='cuda', dtype=torch.int32).to(torch.uint8), decoded,
            num_warps=1, num_stages=1)
        data['all256_decoded_bits'] = [v & 65535 for v in decoded.cpu().view(torch.int16).tolist()]
        assert data['all256_decoded_bits'] == [(((v & 127) << 4) + 14848 if (v & 127) >= 8 else 0) | ((v & 128) << 8) for v in range(256)]
        cases = ((1, 33), (1, 4096), (8, 4096), (16, 4096), (24, 4096),
            (32, 4096), (64, 4096), (64, 5120))
        with torch.inference_mode():
            for case_id, (n, context) in enumerate(cases):
                torch.manual_seed(202610196000 + case_id)
                current_context[0] = context
                lengths = [max(1, context - i * 37 % min(context, 257)) for i in range(n)]
                width = (context + 15) // 16
                pages = n * width
                logical_k = torch.randn(n, width * 16, 2, 128, device='cuda', dtype=torch.bfloat16)
                logical_v = torch.randn_like(logical_k)
                qs = torch.randn(n + 3, 20, 128, device='cuda', dtype=torch.bfloat16)
                q = qs[:n, :16]
                cu = torch.arange(n + 1, device='cuda', dtype=torch.int32)
                used = torch.tensor(lengths, device='cuda', dtype=torch.int32)
                fixtures = []
                shared_table = torch.randperm(pages, device='cuda').int().reshape(n, width)
                for mode in MODES:
                    physical = (pages + 2) * (2 if case_id % 2 else 1)
                    storage, native_view, k, v, layout = allocate(torch, FullAttentionSpec,
                        _reshape_attention_kv_cache, mode, physical, bool(case_id % 2))
                    table = shared_table.clone()
                    out = torch.full((n + 3, 20, 128), 17., device='cuda', dtype=q.dtype)
                    segments = configuration(mode,n,context)['segments']
                    inputs = torch.empty(n + 3, 20, 128, device='cuda', dtype=q.dtype)
                    f = dict(mode=mode, storage=storage, native_view=native_view, k=k, v=v,
                        table=table, output_storage=out, output=out[:, :16], input_storage=inputs,
                        key=inputs[:, 16:18], value=inputs[:, 18:20],
                        slots=torch.full((n + 3,), -1, device='cuda', dtype=torch.int64),
                        segments=segments, layout=layout,
                        partial=torch.full((n, 16, 64, 128), 17., device='cuda', dtype=torch.float32),
                        maximum=torch.full((n, 16, 64), 17., device='cuda', dtype=torch.float32),
                        total=torch.full((n, 16, 64), 17., device='cuda', dtype=torch.float32))
                    fixtures.append(f)

                def refill(f):
                    for src, plane in ((logical_k, f['k']), (logical_v, f['v'])):
                        plane[f['table'].flatten().long()] = (encode(src) if plane.dtype == torch.uint8 else src).reshape(pages, 16, 2, 128)
                    f['expected_storage'] = f['storage'].clone()
                    for i, length in enumerate(lengths):
                        page, token = divmod(length - 1, 16)
                        physical_page = f['table'][i, page]
                        f['key'][i].copy_(logical_k[i, length - 1])
                        f['value'][i].copy_(logical_v[i, length - 1])
                        f['slots'][i] = physical_page.long() * 16 + token
                        f['k'][physical_page, token] = 255 if f['k'].dtype == torch.uint8 else -13.
                        f['v'][physical_page, token] = 255 if f['v'].dtype == torch.uint8 else -13.
                    f['key'][n:].fill_(19.)
                    f['value'][n:].fill_(23.)

                def invoke(f):
                    producer.mode = f['mode']
                    if f['mode'] == 'address32':
                        bounds = assert_safe([q, f['k'], f['v'], f['output'], cu, used, f['table'], f['partial'], f['maximum'], f['total'], scales])
                        data['offset_bounds'][str((n, context))] = bounds
                    writer.write(f['key'], f['value'], f['k'], f['v'], f['slots'])
                    reader.unified_attention(q=q, k=f['k'], v=f['v'], out=f['output'], cu_seqlens_q=cu,
                        max_seqlen_q=1, seqused_k=used, max_seqlen_k=context, softmax_scale=128 ** -.5,
                        causal=True, window_size=(-1, -1), block_table=f['table'], softcap=0.,
                        q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=64,
                        num_par_softmax_segments=f['segments'], softmax_segm_output=f['partial'],
                        softmax_segm_max=f['maximum'], softmax_segm_expsum=f['total'], kv_quant_mode=KVQuantMode.NONE)

                phases = []
                for phase in (0, 1):
                    if phase:
                        q.neg_()
                        logical_k.mul_(.5)
                        logical_v.neg_()
                        lengths = [max(1, length - 7) for length in lengths]
                        used.copy_(torch.tensor(lengths, device='cuda', dtype=torch.int32))
                    phase_checks = []
                    for f in fixtures:
                        refill(f)
                        if phase == 0:
                            for _ in range(3):
                                invoke(f)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                invoke(f)
                            f['graph'] = graph
                        rng = torch.cuda.get_rng_state().clone()
                        saved_inputs = [x.clone() for x in (q, f['key'], f['value'], f['slots'], used, f['table'])]
                        f['graph'].replay()
                        torch.cuda.synchronize()
                        assert torch.equal(torch.cuda.get_rng_state(), rng)
                        assert all(torch.equal(x, y) for x, y in zip(
                            (q, f['key'], f['value'], f['slots'], used, f['table']), saved_inputs))
                        assert torch.equal(f['storage'], f['expected_storage'])
                        assert bool((f['output_storage'][n:] == 17.).all())
                        assert bool((f['output_storage'][:n, 16:] == 17.).all())
                        count = n * 16 * f['segments'] if n <= 64 else 0
                        for field, scale in (('partial', 128), ('maximum', 1), ('total', 1)):
                            assert bool((f[field].flatten()[count * scale:] == 17.).all())
                        expected, cpu_checks = [], []
                        for i, length in enumerate(lengths):
                            keys = logical_k[i, :length]
                            values = logical_v[i, :length]
                            if f['k'].dtype == torch.uint8:
                                keys = decode(encode(keys))
                            if f['v'].dtype == torch.uint8:
                                values = decode(encode(values))
                            keys = keys.float().repeat_interleave(8, 1)
                            values = values.float().repeat_interleave(8, 1)
                            scores = (keys.permute(1, 0, 2) @ q[i].float().unsqueeze(-1)).squeeze(-1) * 128 ** -.5
                            answer = (scores.softmax(-1).unsqueeze(1) @ values.transpose(0, 1)).squeeze(1)
                            expected.append(answer)
                            if i in (0, n // 2, n - 1):
                                for head in (0, 7, 8, 15):
                                    kc, vc, qc = keys[:, head].cpu(), values[:, head].cpu(), q[i, head].cpu().float()
                                    ref = ((kc @ qc) * 128 ** -.5).softmax(0) @ vc
                                    actual = f['output'][i, head].cpu().float()
                                    c = check(actual, ref)
                                    c.update(actual=actual.tolist(), expected=ref.tolist(), sequence=i, head=head)
                                    assert c['passed']
                                    cpu_checks.append(c)
                        c = check(f['output'][:n], torch.stack(expected))
                        assert c['passed'], (case_id, f['mode'], c)
                        phase_checks.append(dict(mode=f['mode'], GPU_reference=c, CPU_references=cpu_checks,
                            all_physical_storage_checked=True, output_and_scratch_guards=True,
                            inputs_unchanged=True, RNG_unchanged=True))
                    assert torch.equal(fixtures[0]['output'], fixtures[1]['output']), 'address32 changed floating output bits'
                    phases.append(dict(phase=phase, outputs_bitwise_same=True, lengths=list(lengths), checks=phase_checks,
                        changed_input_KV_slots_and_lengths=bool(phase)))
                raw_ms = {mode: [] for mode in MODES}
                timing_orders = []
                for trial in range(7):
                    order = fixtures[trial % len(MODES):] + fixtures[:trial % len(MODES)]
                    if trial % 2:order = list(reversed(order))
                    timing_orders.append([f["mode"] for f in order])
                    for f in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):
                            f['graph'].replay()
                        end.record()
                        end.synchronize()
                        raw_ms[f['mode']].append(start.elapsed_time(end) / 20)
                medians = {m: statistics.median(values) for m, values in raw_ms.items()}
                result = dict(batch=n, context=context, strided=bool(case_id % 2), phases=phases,
                    layouts=[f['layout'] for f in fixtures], raw_ms=raw_ms, median_ms=medians,
                    timing_includes_real_writer=True, identical_page_table_values_across_plans=True, timing_plan_orders=timing_orders)
                data['cases'].append(result)
                save()
                print(json.dumps(dict(batch=n, context=context, median_ms=medians)), flush=True)
                del fixtures, f, graph, expected, keys, values, scores, answer
                torch.cuda.empty_cache()
        assert len(data['cases']) == 8
        for path, sha in sources.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
        assert len(data['offset_bounds']) == 8
        data.update(phase='complete', complete=True, full_model_and_Math_not_run=True,
            full_engine_allocator_not_installed=True, prefill_not_tested=True)
    except BaseException as error:
        import traceback
        data.update(phase='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        save()
        signal.alarm(0)


if __name__ == '__main__':
    main()
