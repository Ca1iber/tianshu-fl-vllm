# SPDX-License-Identifier: Apache-2.0
"""Dynamic K int8 + FP32 scales and integer QK decode; independent format references only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

_legacy_path=Path(__file__).resolve().parents[1]/'tianshu_v_int8_20261020/codec.py'
_legacy_spec=importlib.util.spec_from_file_location('_tianshu_dynamic_k_legacy_codec',_legacy_path)
legacy_format_codec=importlib.util.module_from_spec(_legacy_spec)
sys.modules[_legacy_spec.name]=legacy_format_codec
_legacy_spec.loader.exec_module(legacy_format_codec)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reader-source', required=True, type=Path)
    parser.add_argument('--stop-at-utc', required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.stop_at_utc)
    assert deadline.utcoffset() is not None
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 0
    signal.alarm(min(1800, remaining))
    before = subprocess.check_output(['ixsmi'], text=True)
    assert 'Iluvatar BI-V150' in before and 'No running processes found' in before
    root = args.output.resolve()
    assert root.is_relative_to(Path('/workspace/benchmark-results'))
    root.mkdir(parents=True, exist_ok=False)
    (root / 'hardware_before.txt').write_text(before)
    import torch
    import vllm_fl
    import component_launch
    from vllm.platforms import current_platform
    assert current_platform.vendor_name == 'iluvatar'
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    base = Path(__file__).resolve().parent
    parent = base.parent / 'tianshu_k8v16_pipeline_20261019'
    writer_path = base.parent / 'tianshu_soft8_mixed_planes_20261019/writer.py'
    planes = load('_linear_v_control_value_planes', parent / 'value_planes.py')
    control_writer = load('_linear_v_control_writer', writer_path)
    gate = json.loads((base.parent / 'tianshu_v_int8_20261020/GPU_CODECS_GATE_32g.json').read_text())
    assert gate['complete'] and gate['GPU_codecs_match_CPU_contract'] and gate['GPU_K_bits_preserved']
    assert gate['frozen_sources'][str(Path(legacy_format_codec.__file__))] == hashlib.sha256(Path(legacy_format_codec.__file__).read_bytes()).hexdigest()
    prior = legacy_format_codec._prior_codec()
    dynamic_codec=load('_tianshu_dynamicK_CPU_contract',base/'codec.py')
    dynamic_source=load('_tianshu_dynamicK_reader_generator',base/'reader_source.py')
    dynamic_writer=load('_tianshu_dynamicK_writer',base/'writer.py')
    codec_gate_path=Path('/workspace/benchmark-results/20261020_dynamic_k8_gpu_codecs_tianshu/result.json')
    codec_gate=json.loads(codec_gate_path.read_text());assert codec_gate['complete']
    assert codec_gate['frozen_sources'][str(base/'writer.py')]==hashlib.sha256((base/'writer.py').read_bytes()).hexdigest()
    warp_gate_path=base.parent/'tianshu_int8_dot_capability_20261020/GPU_WARP1_GATE_32g.json'
    warp_gate=json.loads(warp_gate_path.read_text());assert warp_gate['complete'] and warp_gate['successful_shapes']==2
    reader_gate=json.loads((base/'CPU_READER_GATE_32g.json').read_text());assert reader_gate['complete']
    for path,expected in reader_gate['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    models = []
    for dynamic in (False,True):
        reader,proof=prior.load_codec(args.reader_source,root/f'control_base_{dynamic}')
        assert proof['generated_sha256']==dynamic_source.PARENT_SHA
        writer=control_writer
        if dynamic:
            reader,newproof=dynamic_source.load(Path(reader.__file__),root/'dynamic_reader')
            assert newproof['generated_sha256']==reader_gate['generated_source_sha256']
            proof=dict(base_codec=proof,dynamic=newproof);writer=dynamic_writer
        models.append(dict(label='dynamic_K8_INT8_QK' if dynamic else 'qualified_K8V16',dynamic=dynamic,
                           denominator=None,page_bytes=12416 if dynamic else 12288,reader=reader,writer=writer,proof=proof))
    for model in models:
        model['invoke'], model['producer'] = component_launch.attach(model['reader'])
    sources = [base/'codec.py',base/'reader_source.py',base/'writer.py',base/'CPU_READER_GATE_32g.json',codec_gate_path,warp_gate_path,Path(__file__), Path(legacy_format_codec.__file__), Path(component_launch.__file__),
               component_launch.LEGACY, component_launch.POLICY, args.reader_source, writer_path,
               Path(prior.__file__), Path(prior.legacy.__file__), parent / 'value_planes.py',
               base.parent / 'tianshu_v_int8_20261020/GPU_CODECS_GATE_32g.json']
    sources += [Path(model['reader'].__file__) for model in models]
    sources += [Path(model['writer'].__file__) for model in models]
    frozen = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for path in sources:
        copied = root / 'source_snapshot' / path.resolve().relative_to('/')
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(path.read_bytes())
    result = dict(complete=False, phase='decode component', runner_pid=os.getpid(),
                  started_utc=datetime.now(timezone.utc).isoformat(), frozen_sources=frozen,
                  launch_source_proof=component_launch.source_proof(), cases=[],
                  model_sources={model['label']: model['proof'] for model in models},
                  control='qualified K8V16 reader and launch policy, 12288-byte physical pages',
                  candidate='per-token/head FP32 K scales, signed K8 and row-quantized Q with INT32 QK; BF16 V unchanged',
                  engine_allocator_connected=False, prefill_executed=False, model_executed=False,
                  service_executed=False, official_benchmark=False, adoption_allowed=False,
                  torch_version=torch.__version__, GPU=torch.cuda.get_device_name(0),
                  time_scope='hot CUDA Graph: one KV update and one paged decode layer; 9 alternating trials x 20 replays')
    def save():
        temporary = root / 'result.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(root / 'result.json')
    save()
    def quantize_k(x):
        bits = x.view(torch.int16).to(torch.int32)
        magnitude = bits & 32767
        rounded = ((magnitude + 7 + ((magnitude >> 4) & 1)) >> 4) - 928
        return (torch.where(rounded >= 8, rounded.clamp(max=127), 0) | ((bits & 32768) >> 8)).to(torch.uint8)
    def dequantize_k(x):
        packed = x.to(torch.int32)
        magnitude = packed & 127
        return (torch.where(magnitude >= 8, (magnitude << 4) + 14848, 0) | ((packed & 128) << 8)).to(torch.int16).view(torch.bfloat16)
    def error(actual, expected):
        actual, expected = actual.float(), expected.float()
        delta = (actual - expected).abs()
        bad = delta > .02 + .01 * expected.abs()
        return dict(passed=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all() and not bad.any()),
                    max_abs=delta.max().item(), failed_elements=bad.sum().item(), atol=.02, rtol=.01)
    def reference(q, k, v, lengths):
        n, width = k.shape[:2]
        query = q.float().reshape(n, 2, 8, 128)
        scores = torch.einsum('bhgd,bkhd->bhgk', query, k.float()) * (128 ** -.5)
        mask = torch.arange(width, device=q.device)[None, :] < lengths[:, None]
        scores.masked_fill_(~mask[:, None, None, :], -float('inf'))
        return torch.einsum('bhgk,bkhd->bhgd', scores.softmax(-1), v.float()).reshape(n, 16, 128)
    def cpu_checks(q, k, v, lengths, expected):
        checks = []
        for sequence in sorted({0, q.shape[0] // 2, q.shape[0] - 1}):
            length = lengths[sequence]
            key = k[sequence, :length].cpu().float()
            value = v[sequence, :length].cpu().float()
            for head in (0, 7, 8, 15):
                query = q[sequence, head].cpu().float()
                scores = (key[:, head // 8] @ query) * (128 ** -.5)
                answer = scores.softmax(0) @ value[:, head // 8]
                checks.append(error(expected[sequence, head].cpu(), answer))
        assert all(check['passed'] for check in checks)
        return checks
    cases = [(f'decode{n}_{length}', [length] * n, False)
             for n in (1,8,32,64) for length in (4096,5120)]
    cases += [('decode8_ragged_strided', [33, 65, 127, 512, 2048, 4096, 5000, 5120], True),
              ('decode64_ragged_strided', [4096 - (i % 7) * 131 for i in range(64)], True)]
    cases += [('decode40_16384',[16384]*40,False)]
    try:
        with torch.inference_mode():
            for case_id, (name, lengths, strided) in enumerate(cases):
                seed = 2026102000 + case_id
                torch.manual_seed(seed)
                n = len(lengths)
                width = (max(lengths) + 15) // 16
                pages = n * width
                logical_k = torch.randn(n, width * 16, 2, 128, device='cuda', dtype=torch.bfloat16)
                logical_v = torch.randn_like(logical_k)
                q = torch.randn(n, 20 if strided else 16, 128, device='cuda', dtype=torch.bfloat16)[:, :16]
                table = torch.zeros((n, width + 3 if strided else width), device='cuda', dtype=torch.int32)[:, :width]
                table.copy_(torch.randperm(pages, device='cuda').int().reshape(n, width))
                used = torch.tensor(lengths, device='cuda', dtype=torch.int32)
                cu = torch.arange(n + 1, device='cuda', dtype=torch.int32)
                fixtures = []
                for model in models:
                    denominator = model['denominator']
                    page_bytes = model['page_bytes']
                    physical_pages = (pages + 2) * (2 if strided else 1)
                    raw = torch.full((physical_pages * page_bytes,), 17, device='cuda', dtype=torch.uint8)
                    full = torch.as_strided(raw, (physical_pages, 2, 16, 2, 128), (page_bytes, 4096, 256, 128, 1))
                    kv = full[::2] if strided else full
                    k, v = kv.unbind(1)
                    if denominator is None:
                        v = planes.value_plane(v)
                    scale_cache = torch.as_strided(raw.view(torch.float32),(physical_pages,16,2),(page_bytes//4,2,1),12288//4)[::2 if strided else 1] if model['dynamic'] else None
                    out_storage = torch.full((n + 13, 20 if strided else 16, 128), 17., device='cuda', dtype=torch.bfloat16)
                    out = out_storage[:n, :16]
                    kw = dict(q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=1,
                              seqused_k=used, max_seqlen_k=width * 16, softmax_scale=128 ** -.5,
                              causal=True, window_size=(-1, -1), block_table=table, softcap=0.,
                              q_descale=torch.ones(1, device='cuda'), k_descale=torch.ones(1, device='cuda'),
                              v_descale=torch.ones(1, device='cuda'), seq_threshold_3D=64,
                              num_par_softmax_segments=64,
                              softmax_segm_output=torch.full((64, 16, 64, 128), 17., device='cuda', dtype=torch.float32),
                              softmax_segm_max=torch.full((64, 16, 64), 17., device='cuda', dtype=torch.float32),
                              softmax_segm_expsum=torch.full((64, 16, 64), 17., device='cuda', dtype=torch.float32))
                    fixtures.append(dict(model=model, raw=raw, k=k, v=v, kv=kv, scale_cache=scale_cache, out=out, out_storage=out_storage,
                                         kw=kw, page_bytes=page_bytes, graph=None, slots=None, key=None, value=None))
                phase_results = []
                for phase in range(2):
                    if phase:
                        logical_k.mul_(.5).add_(.0625)
                        logical_v.neg_()
                        q.mul_(-.75).add_(.125)
                        lengths = [length - min(7, length - 1) for length in lengths]
                        used.copy_(torch.tensor(lengths, device='cuda', dtype=torch.int32))
                        table.copy_(torch.randperm(pages, device='cuda').int().reshape(n, width))
                    control_encoded_k = quantize_k(logical_k)
                    control_reference_k = dequantize_k(control_encoded_k)
                    checks = []
                    for fixture in fixtures:
                        denominator = fixture['model']['denominator']
                        if fixture['model']['dynamic']:
                            host_key=logical_k.cpu();host_q=q.cpu()
                            codes,scales=dynamic_codec.encode_cpu(host_key)
                            qcodes,qscales=dynamic_codec.encode_cpu(host_q)
                            encoded_k=codes.cuda()
                            reference_k=(codes.view(torch.int8).float()*scales[...,None]).cuda()
                            reference_q=(qcodes.view(torch.int8).float()*qscales[...,None]).cuda()
                            scale_values=scales.reshape(pages,16,2).cuda()
                            del host_key,host_q,codes,scales,qcodes,qscales
                        else:
                            encoded_k=control_encoded_k;reference_k=control_reference_k;reference_q=q
                        encoded_v = logical_v
                        reference_v = logical_v
                        ids = table.flatten().long()
                        fixture['k'][ids] = encoded_k.reshape(pages, 16, 2, 128)
                        fixture['v'][ids] = encoded_v.reshape(pages, 16, 2, 128)
                        if fixture['model']['dynamic']:fixture['scale_cache'][ids]=scale_values
                        expected_storage = fixture['raw'].clone()
                        positions = used.long() - 1
                        slots = table[torch.arange(n, device='cuda'), positions // 16].long() * 16 + positions % 16
                        slots = torch.cat((slots, torch.full((3,), -1, device='cuda', dtype=torch.int64)))
                        key = torch.cat((logical_k[torch.arange(n, device='cuda'), positions], torch.full((3, 2, 128), 19., device='cuda', dtype=torch.bfloat16)))
                        value = torch.cat((logical_v[torch.arange(n, device='cuda'), positions], torch.full((3, 2, 128), 23., device='cuda', dtype=torch.bfloat16)))
                        if fixture['slots'] is None:
                            fixture.update(slots=slots, key=key, value=value)
                        else:
                            fixture['slots'].copy_(slots)
                            fixture['key'].copy_(key)
                            fixture['value'].copy_(value)
                        valid_slots = slots[:n]
                        fixture['k'][valid_slots // 16, valid_slots % 16] = 255
                        if fixture['model']['dynamic']:fixture['scale_cache'][valid_slots//16,valid_slots%16]=-3.
                        fixture['v'][valid_slots // 16, valid_slots % 16] = -13. if denominator is None else 255
                        def invoke(f=fixture):
                            f['model']['writer'].write(f['key'], f['value'], f['k'], f['v'], f['slots'])
                            f['model']['invoke'](f['kw'])
                        if not phase:
                            for _ in range(3):
                                invoke()
                            fixture['graph'] = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(fixture['graph']):
                                invoke()
                        rng_before = torch.cuda.get_rng_state().clone()
                        fixture['graph'].replay()
                        torch.cuda.synchronize()
                        assert torch.equal(torch.cuda.get_rng_state(), rng_before)
                        assert torch.equal(fixture['raw'], expected_storage), 'whole cache storage mismatch after Graph writer'
                        assert bool((fixture['out_storage'][n:] == 17.).all())
                        if strided:
                            assert bool((fixture['out_storage'][:n, 16:] == 17.).all())
                        segments = component_launch.choose_config(n, width * 16)['segments']
                        for field, factor in (('softmax_segm_output', 128), ('softmax_segm_max', 1), ('softmax_segm_expsum', 1)):
                            assert bool((fixture['kw'][field].flatten()[n * 16 * segments * factor:] == 17.).all())
                        expected = reference(reference_q, reference_k, reference_v, used)
                        numerical = error(fixture['out'], expected)
                        assert numerical['passed'], numerical
                        cpu = cpu_checks(reference_q, reference_k, reference_v, lengths, expected)
                        approximation = (reference_v.float() - logical_v.float()).abs()
                        saturation = None
                        checks.append(dict(label=fixture['model']['label'], numerical=numerical, cpu_checks=cpu,
                                           V_max_quantization_abs=approximation.max().item(),
                                           V_random_fixture_out_of_range=saturation, V_fixture_elements=logical_v.numel(),
                                           effective_Q_K_from_independent_CPU_codec=fixture['model']['dynamic'],
                                           physical_page_bytes=fixture['page_bytes'], raw_cache_bytes=fixture['raw'].numel(),
                                           all_storage_bitwise_exact=True, output_and_segment_guards=True, RNG_unchanged=True))
                    phase_results.append(dict(phase=phase, lengths=list(lengths), checks=checks,
                                              Q_K_V_table_lengths_slots_changed_in_place=bool(phase)))
                    torch.save(dict(q=q.cpu(), table=table.cpu(), lengths=used.cpu(), seed=seed),
                               root / f'{name}_phase{phase}_small_inputs.pt')
                measurements = [[] for _ in fixtures]
                for trial in range(9):
                    order = range(len(fixtures)) if trial % 2 == 0 else reversed(range(len(fixtures)))
                    for index in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):
                            fixtures[index]['graph'].replay()
                        end.record()
                        end.synchronize()
                        measurements[index].append(start.elapsed_time(end) / 20)
                medians = list(map(statistics.median, measurements))
                current = dict(name=name, seed=seed, strided=strided, batch=n, phases=phase_results,
                               labels=[model['label'] for model in models], raw_ms=measurements, median_ms=medians,
                               candidate_speedup_percent=[100 * (medians[0] / value - 1) for value in medians[1:]],
                               actual_launch=[model['producer'].last_record for model in models],
                               reference_scope='CPU selected sequences/heads plus full GPU FP32 grouped-GQA reference')
                result['cases'].append(current)
                save()
                print(json.dumps({key: current[key] for key in ('name', 'median_ms', 'candidate_speedup_percent')}), flush=True)
                planes.clear_cache()
                del fixtures, expected_storage, reference_v, encoded_v, approximation,reference_k,reference_q,encoded_k
                torch.cuda.empty_cache()
            assert len(result['cases']) == len(cases)
            assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in frozen.items())
            result.update(complete=True, phase='complete', numerical_passed=True,
                          same_Graph_Q_K_V_table_lengths_verified=True,
                          ended_utc=datetime.now(timezone.utc).isoformat())
            save()
    except BaseException as error:
        import traceback
        result.update(phase='failed', error=repr(error), traceback=traceback.format_exc(), ended_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise
    finally:
        (root / 'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'], text=True))


if __name__ == '__main__':
    main()
