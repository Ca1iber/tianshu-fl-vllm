# SPDX-License-Identifier: Apache-2.0
"""Native FP16 V versus BF16 V with identical page size; no model hooks."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import codec
import launch_sweep
import domain_gpu


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-at-utc', required=True)
    args = parser.parse_args()
    remaining = int((datetime.fromisoformat(args.stop_at_utc) - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 1200
    signal.alarm(min(1800, remaining))
    before = subprocess.check_output(['ixsmi'], text=True)
    assert 'Iluvatar BI-V150' in before and 'No running processes found' in before
    root = args.output.resolve()
    assert root.is_relative_to(Path('/workspace/benchmark-results'))
    root.mkdir(parents=True, exist_ok=False)
    (root / 'hardware_before.txt').write_text(before)
    base = Path(__file__).resolve().parent
    gate = json.loads((base / 'CPU_GATE_32g.json').read_text())
    assert gate['complete'] and gate['CPU_only']
    for path, sha in gate['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
    import torch
    import triton
    import vllm_fl
    from vllm.platforms import current_platform
    assert current_platform.vendor_name == 'iluvatar' and torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    launch_dir = base.parent / 'tianshu_k8v16_address32_20261020'
    sys.path.insert(0, str(launch_dir))
    component_launch = load("_nativef16_component_launch", base / "component_launch.py")
    planes = load('_e5_current_value_planes', base.parent / 'tianshu_k8v16_pipeline_20261019/value_planes.py')
    candidate_planes = load("_nativef16_value_planes", base / "value_planes.py")
    control_writer = load('_e5_control_writer', codec.WRITER)
    models = [dict(label='control', page_bytes=12288,
                   reader=load('_e5_control_reader', codec.READER), writer=control_writer)]
    generated = root / 'generated'
    generated.mkdir()
    text, writer_proof = codec.render_writer(codec.WRITER.read_bytes())
    writer_path = generated / 'writer.py'
    writer_path.write_text(text)
    candidate_writer = load('_e5_candidate_writer', writer_path)
    proofs = {'writer': writer_proof}
    for config in launch_sweep.CONFIGS:
        mode=config["label"]
        text, proofs[mode] = codec.render_reader(codec.READER.read_bytes(), "FP16_PV")
        path = generated / f'reader_{mode}.py'
        path.write_text(text)
        models.append(dict(label=mode, page_bytes=12288, config=config,
                           reader=load('_e5_reader_' + mode, path), writer=candidate_writer))
    for model in models:
        model['invoke'], model['producer'] = launch_sweep.attach(model,component_launch,root/"codegen")
    paths = [Path(__file__).resolve(), Path(launch_sweep.__file__), base / 'CPU_GATE_32g.json',
             Path(component_launch.__file__), component_launch.LEGACY, component_launch.POLICY,
             launch_dir / 'guard.py', launch_dir / 'source.py', Path(planes.__file__), Path(candidate_planes.__file__)]
    paths += [Path(p) for p in gate['sources']]
    paths += [Path(m['reader'].__file__) for m in models] + [writer_path]
    paths += [Path(sys.modules[n].__file__) for n in ('torch','triton')]
    frozen = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for path, sha in frozen.items():
        dest = root / 'source_snapshot' / Path(path).relative_to('/')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(path).read_bytes())
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == sha
    result = dict(complete=False, phase='codec validation', runner_pid=__import__('os').getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(), frozen_sources=frozen,
        source_proofs=proofs, launch_proof=component_launch.source_proof(),
        torch_version=torch.__version__, triton_version=triton.__version__,
        GPU=torch.cuda.get_device_name(0), cases=[], codec_validation={}, declared_grid=launch_sweep.CONFIGS,
        compiled_kernel_records={m["label"]:m["producer"]._sweep_codegen.records for m in models},
        engine_allocator_connected=False, prefill_executed=False, model_executed=False,
        service_executed=False, official_benchmark=False, adopted=False,
        time_scope='one KV writer plus one paged decode layer; 9 alternating trials x 20 Graph replays')
    def save():
        tmp = root / 'result.tmp'
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        tmp.replace(root / 'result.json')
    def error(actual, expected, atol=.002, rtol=.01):
        actual, expected = (actual.double(), expected.double()) if actual.device.type == "cpu" else (actual.float(), expected.float())
        delta = (actual - expected).abs()
        failed = delta > atol + rtol * expected.abs()
        return dict(passed=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all() and not failed.any()),
                    max_abs=delta.max().item(), failed_elements=int(failed.sum()), atol=atol, rtol=rtol)
    def quantize_k(x):
        bits = x.view(torch.int16).int()
        mag = bits & 32767
        code = ((mag + 7 + ((mag >> 4) & 1)) >> 4) - 928
        return (torch.where(code >= 8, code.clamp(max=127), 0) | ((bits & 32768) >> 8)).byte()
    def decode_k(x):
        bits = x.int(); mag = bits & 127
        return (torch.where(mag >= 8, (mag << 4) + 14848, 0) | ((bits & 128) << 8)).short().view(torch.bfloat16)
    def quantize_v(x):
        return x.half()
    def reference(q, k, v, lengths):
        n, width = k.shape[:2]
        query = q.float().reshape(n, 2, 8, 128)
        scores = torch.einsum('bhgd,bkhd->bhgk', query, k.float()) * (128 ** -.5)
        mask = torch.arange(width, device=q.device)[None, :] < lengths[:, None]
        scores.masked_fill_(~mask[:, None, None, :], -float('inf'))
        return torch.einsum('bhgk,bkhd->bhgd', scores.softmax(-1), v.float()).reshape(n, 16, 128)
    def cpu_checks(q, k, v, lengths, actual):
        records = []
        for sequence in sorted({0, q.shape[0] // 2, q.shape[0] - 1}):
            length = lengths[sequence]
            key = k[sequence, :length].cpu().double()
            value = v[sequence, :length].cpu().double()
            for head in (0, 7, 8, 15):
                query = q[sequence, head].cpu().double()
                answer = ((key[:, head // 8] @ query) * (128 ** -.5)).softmax(0) @ value[:, head // 8]
                check = error(actual[sequence, head].cpu(), answer)
                check.update(sequence=sequence, head=head, scalars=128)
                assert check['passed'], check
                records.append(check)
        return records
    save()
    try:
        with torch.inference_mode():
            result["codec_validation"] = domain_gpu.validate(codec, models, candidate_writer, control_writer, torch)
            result["phase"] = "writer plus decode component"; save()
            cases = [(f'decode{n}_{length}', [length] * n, False)
                     for n in (64,) for length in (4096, 5120)]
            cases += [("decode64_ragged_strided", [4096-(i%7)*131 for i in range(64)], True)]
            for case_id, (name, lengths, strided) in enumerate(cases):
                seed = 2026102400 + case_id
                torch.manual_seed(seed)
                n = len(lengths); width = (max(lengths) + 15) // 16; pages = n * width
                logical_k = torch.randn(n, width*16, 2, 128, device='cuda', dtype=torch.bfloat16)
                logical_v = torch.randn_like(logical_k)
                q_storage = torch.randn(n, 20 if strided else 16, 128, device='cuda', dtype=torch.bfloat16)
                q = q_storage[:, :16]
                table_storage = torch.zeros((n, width+3 if strided else width), device='cuda', dtype=torch.int32)
                table = table_storage[:, :width]
                used = torch.tensor(lengths, device='cuda', dtype=torch.int32)
                cu = torch.arange(n+1, device='cuda', dtype=torch.int32)
                table.copy_(torch.randperm(pages, device='cuda').int().reshape(n, width))
                fixtures = []
                for model in models:
                    physical_pages = (pages+2) * (2 if strided else 1)
                    raw = torch.full((physical_pages*model['page_bytes'],), 17, device='cuda', dtype=torch.uint8)
                    full = torch.as_strided(raw, (physical_pages,2,16,2,128), (model['page_bytes'],4096,256,128,1))
                    kv = full[::2] if strided else full
                    k,v = kv.unbind(1)
                    v = planes.value_plane(v) if model['label']=='control' else candidate_planes.value_plane(v)
                    out_storage = torch.full((n+13,20 if strided else 16,128), 17., device='cuda', dtype=torch.bfloat16)
                    out = out_storage[:n,:16]
                    kw = dict(q=q,k=k,v=v,out=out,cu_seqlens_q=cu,max_seqlen_q=1,seqused_k=used,
                        max_seqlen_k=width*16,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),
                        block_table=table,softcap=0.,q_descale=torch.ones(1,device='cuda'),
                        k_descale=torch.ones(1,device='cuda'),v_descale=torch.ones(1,device='cuda'),
                        seq_threshold_3D=64,num_par_softmax_segments=64,
                        softmax_segm_output=torch.full((64,16,64,128),17.,device='cuda'),
                        softmax_segm_max=torch.full((64,16,64),17.,device='cuda'),
                        softmax_segm_expsum=torch.full((64,16,64),17.,device='cuda'))
                    fixtures.append(dict(model=model,raw=raw,k=k,v=v,out=out,out_storage=out_storage,
                                         kw=kw,graph=None,slots=None,key=None,value=None))
                phase_records = []
                for phase in range(2):
                    if phase:
                        logical_k.mul_(.5); logical_v.neg_(); q.neg_()
                        lengths = [length-min(7,length-1) for length in lengths]
                        used.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
                        table.copy_(torch.randperm(pages,device='cuda').int().reshape(n,width))
                    input_tensors = [logical_k, logical_v, q_storage, table_storage, used]
                    input_backups = [x.clone() for x in input_tensors]
                    encoded_k = quantize_k(logical_k); reference_k = decode_k(encoded_k)
                    encoded_v = quantize_v(logical_v); reference_v = encoded_v
                    checks = []
                    for fixture in fixtures:
                        candidate = fixture['model']['label'] != 'control'
                        stored_v = encoded_v if candidate else logical_v
                        answer_v = reference_v if candidate else logical_v
                        ids = table.flatten().long()
                        fixture['k'][ids] = encoded_k.reshape(pages,16,2,128)
                        fixture['v'][ids] = stored_v.reshape(pages,16,2,128)
                        expected_storage = fixture['raw'].clone()
                        positions = used.long()-1
                        slots = table[torch.arange(n,device='cuda'),positions//16].long()*16+positions%16
                        slots = torch.cat((slots,torch.full((3,),-1,device='cuda',dtype=torch.int64)))
                        key = torch.cat((logical_k[torch.arange(n,device='cuda'),positions],
                                         torch.full((3,2,128),19.,device='cuda',dtype=torch.bfloat16)))
                        value = torch.cat((logical_v[torch.arange(n,device='cuda'),positions],
                                           torch.full((3,2,128),23.,device='cuda',dtype=torch.bfloat16)))
                        if fixture['slots'] is None: fixture.update(slots=slots,key=key,value=value)
                        else:
                            fixture['slots'].copy_(slots);fixture['key'].copy_(key);fixture['value'].copy_(value)
                        fixture['k'][slots[:n]//16,slots[:n]%16] = 255
                        fixture['v'][slots[:n]//16,slots[:n]%16] = 255 if candidate else -13.
                        def invoke(f=fixture):
                            f['model']['writer'].write(f['key'],f['value'],f['k'],f['v'],f['slots'])
                            f['model']['invoke'](f['kw'])
                        if not phase:
                            for _ in range(3): invoke()
                            fixture['graph'] = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(fixture['graph']): invoke()
                        fixture_inputs = [fixture["slots"], fixture["key"], fixture["value"]]
                        fixture_backups = [x.clone() for x in fixture_inputs]
                        rng = torch.cuda.get_rng_state().clone()
                        fixture['graph'].replay(); torch.cuda.synchronize()
                        assert torch.equal(torch.cuda.get_rng_state(),rng)
                        assert torch.equal(fixture['raw'],expected_storage)
                        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a,b in zip(input_tensors,input_backups))
                        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a,b in zip(fixture_inputs,fixture_backups))
                        assert bool((fixture['out_storage'][n:] == 17.).all())
                        if strided: assert bool((fixture['out_storage'][:n,16:] == 17.).all())
                        segments = fixture["model"]["producer"].last_record["segments"]
                        for field,factor in (('softmax_segm_output',128),('softmax_segm_max',1),('softmax_segm_expsum',1)):
                            assert bool((fixture['kw'][field].flatten()[n*16*segments*factor:] == 17.).all())
                        expected = reference(q,reference_k,answer_v,used)
                        numerical = error(fixture['out'],expected)
                        assert numerical['passed'], (name,fixture['model']['label'],numerical)
                        cpu = cpu_checks(q,reference_k,answer_v,lengths,fixture['out'])
                        checks.append(dict(label=fixture['model']['label'],numerical=numerical,CPU_FP64=cpu,
                            full_cache_storage_bitwise_exact=True,output_and_scratch_guards=True,RNG_unchanged=True,
                            full_inputs_bitwise_unchanged=True,
                            quantized_V_reference=candidate,physical_page_bytes=fixture['model']['page_bytes'],
                            V_quantization_max_abs=(answer_v.float()-logical_v.float()).abs().max().item()))
                    phase_records.append(dict(phase=phase,lengths=list(lengths),checks=checks,
                        same_Graph_Q_K_V_table_lengths_slots_changed=bool(phase)))
                measurements = [[] for _ in fixtures]
                for trial in range(9):
                    order = range(len(fixtures)) if trial%2 == 0 else reversed(range(len(fixtures)))
                    for index in order:
                        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20): fixtures[index]['graph'].replay()
                        end.record();end.synchronize()
                        measurements[index].append(start.elapsed_time(end)/20)
                medians = list(map(statistics.median,measurements))
                record = dict(name=name,seed=seed,strided=strided,batch=n,phases=phase_records,
                    labels=[m['label'] for m in models],raw_ms=measurements,median_ms=medians,
                    candidate_time_reduction_percent=[100*(1-ms/medians[0]) for ms in medians[1:]],
                    actual_launch=[m['producer'].last_record for m in models])
                result['cases'].append(record);save()
                print(json.dumps({k:record[k] for k in ('name','median_ms','candidate_time_reduction_percent')}),flush=True)
                planes.clear_cache(); candidate_planes.clear_cache()
                del fixtures,expected_storage,expected,logical_k,logical_v,reference_k,reference_v,encoded_k,encoded_v,answer_v,stored_v
                torch.cuda.empty_cache()
            assert len(result['cases']) == len(cases)
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p,sha in frozen.items())
            result.update(phase='complete',complete=True,ended_utc=datetime.now(timezone.utc).isoformat(),
                full_GPU_FP32_and_selected_CPU_FP64_references_passed=True)
            save()
    except BaseException as exc:
        import traceback
        result.update(phase='failed',error=repr(exc),traceback=traceback.format_exc(),ended_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise
    finally:
        (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))


if __name__ == '__main__': main()
