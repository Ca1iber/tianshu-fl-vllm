# SPDX-License-Identifier: Apache-2.0
"""Deterministic B64/4K layer workload for IXplorer; timings are profiler-perturbed."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('control', 'linear', 'magic'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reader-source', required=True, type=Path)
    parser.add_argument('--stop-at-utc', required=True)
    args = parser.parse_args()
    remaining = int((datetime.fromisoformat(args.stop_at_utc) - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 0
    signal.alarm(min(600, remaining))
    root = args.output.resolve()
    assert root.is_relative_to(Path('/workspace/benchmark-results'))
    root.mkdir(parents=True, exist_ok=False)
    import torch
    import vllm_fl
    import component_launch
    from vllm.platforms import current_platform
    assert current_platform.vendor_name == 'iluvatar' and torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    base = Path(__file__).resolve().parent
    codec_folder = base.parent / ('tianshu_v_int8_magic_20261020' if args.mode == 'magic' else 'tianshu_v_int8_20261020')
    codec = load('_profile_owned_codec', codec_folder / 'codec.py')
    writer_path = base.parent / 'tianshu_soft8_mixed_planes_20261019/writer.py'
    gate = json.loads((codec_folder / 'GPU_CODECS_GATE_32g.json').read_text())
    assert gate['complete'] and gate['GPU_codecs_match_CPU_contract']
    assert gate['frozen_sources'][str(Path(codec.__file__))] == hashlib.sha256(Path(codec.__file__).read_bytes()).hexdigest()
    if args.mode == 'control':
        reader, proof = codec._prior_codec().load_codec(args.reader_source, root / 'reader')
        assert proof['generated_sha256'] == 'dc69f525d8c7686a8977f50e901af39820ce92486894890631c53fc01dc46700'
    else:
        reader, unused_writer, proof = codec.load_modules(args.reader_source, writer_path, root / 'reader', 32)
        expected = next(case['source_proof'] for case in gate['cases'] if case['denominator'] == 32)
        assert proof['reader']['generated_sha256'] == expected['reader']['generated_sha256']
    invoke, producer = component_launch.attach(reader)
    torch.manual_seed(2026102004)
    n, width = 64, 256
    pages = n * width
    logical_k = torch.randn(n, width * 16, 2, 128, device='cuda', dtype=torch.bfloat16)
    bits = logical_k.view(torch.int16).to(torch.int32)
    magnitude = bits & 32767
    rounded = ((magnitude + 7 + ((magnitude >> 4) & 1)) >> 4) - 928
    encoded_k = (torch.where(rounded >= 8, rounded.clamp(max=127), 0) | ((bits & 32768) >> 8)).to(torch.uint8)
    codes_v = torch.randint(0, 256, logical_k.shape, device='cuda', dtype=torch.int32).to(torch.uint8)
    signed_v = torch.where(codes_v.to(torch.int32) < 128, codes_v.to(torch.int32), codes_v.to(torch.int32) - 256)
    logical_v = (signed_v.float() / 32).to(torch.bfloat16)
    q = torch.randn(n, 16, 128, device='cuda', dtype=torch.bfloat16)
    table = torch.randperm(pages, device='cuda').int().reshape(n, width)
    page_bytes = 12288 if args.mode == 'control' else 8192
    raw = torch.full(((pages + 2) * page_bytes,), 17, device='cuda', dtype=torch.uint8)
    kv = torch.as_strided(raw, (pages + 2, 2, 16, 2, 128), (page_bytes, 4096, 256, 128, 1))
    k, v = kv.unbind(1)
    if args.mode == 'control':
        planes = load('_profile_control_planes', base.parent / 'tianshu_k8v16_pipeline_20261019/value_planes.py')
        v = planes.value_plane(v)
    ids = table.flatten().long()
    k[ids] = encoded_k.reshape(pages, 16, 2, 128)
    v[ids] = (logical_v if args.mode == 'control' else codes_v).reshape(pages, 16, 2, 128)
    out_storage = torch.full((n + 13, 16, 128), 17., device='cuda', dtype=torch.bfloat16)
    kw = dict(q=q, k=k, v=v, out=out_storage[:n], cu_seqlens_q=torch.arange(n+1, device='cuda', dtype=torch.int32),
              max_seqlen_q=1, seqused_k=torch.full((n,), 4096, device='cuda', dtype=torch.int32),
              max_seqlen_k=4096, softmax_scale=128**-.5, causal=True, window_size=(-1,-1),
              block_table=table, softcap=0., q_descale=torch.ones(1, device='cuda'),
              k_descale=torch.ones(1, device='cuda'), v_descale=torch.ones(1, device='cuda'),
              seq_threshold_3D=64, num_par_softmax_segments=64,
              softmax_segm_output=torch.full((64,16,64,128),17.,device='cuda',dtype=torch.float32),
              softmax_segm_max=torch.full((64,16,64),17.,device='cuda',dtype=torch.float32),
              softmax_segm_expsum=torch.full((64,16,64),17.,device='cuda',dtype=torch.float32))
    sources = [Path(__file__), Path(component_launch.__file__), component_launch.LEGACY,
               component_launch.POLICY, Path(codec.__file__), args.reader_source, Path(reader.__file__)]
    if hasattr(codec, 'PARENT'):
        sources.append(codec.PARENT)
    frozen = {str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for path in sources:
        target = root / 'source_snapshot' / path.resolve().relative_to('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    for _ in range(3):
        invoke(kw)
    torch.cuda.synchronize()
    # With --launch-skip 3 --launch-count 1, IXplorer collects the first call below.
    for _ in range(8):
        invoke(kw)
    torch.cuda.synchronize()
    assert torch.isfinite(out_storage[:n]).all() and bool((out_storage[n:] == 17.).all())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in frozen.items())
    result = dict(complete=True, mode=args.mode, GPU=torch.cuda.get_device_name(0), batch=n,
                  actual_length=4096, page_bytes=page_bytes, actual_launch=producer.last_record,
                  source_proof=proof, launch_source_proof=component_launch.source_proof(),
                  frozen_sources=frozen, calls=11, intended_profiled_call=4,
                  same_logical_V_all_modes=True, output_finite_and_tail_preserved=True,
                  sampling_model_service_not_executed=True, profile_workload_only=True,
                  not_an_unprofiled_timing_result=True, adoption_allowed=False,
                  ended_utc=datetime.now(timezone.utc).isoformat())
    (root / 'result.json').write_text(json.dumps(result,indent=2)+'\n')
    torch.save(dict(output=out_storage[:n].cpu(),q=q.cpu(),table=table.cpu()),root/'small_inputs_output.pt')
    print(json.dumps(dict(complete=True,mode=args.mode,launch=producer.last_record,scope='hardware-profile workload only')))


if __name__=='__main__':
    main()
