# SPDX-License-Identifier: Apache-2.0
"""GPU codec/writer gate only. Does not test attention or install engine hooks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

import codec


def equal_bytes(actual, expected, label):
    actual = bytes(actual.cpu().tolist())
    expected = bytes(expected)
    if actual != expected:
        first = next(i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
        raise AssertionError((label, 'first mismatched byte', first, actual[first], expected[first]))
    return hashlib.sha256(actual).hexdigest()


def metadata(compiled):
    return dict(num_warps=compiled.metadata.num_warps,
                num_stages=compiled.metadata.num_stages,
                n_regs=getattr(compiled, 'n_regs', None),
                n_spills=getattr(compiled, 'n_spills', None))


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
    root = args.output.resolve()
    assert root.is_relative_to(Path('/workspace/benchmark-results'))
    before = subprocess.check_output(['ixsmi'], text=True)
    assert 'Iluvatar BI-V150' in before and 'No running processes found' in before
    root.mkdir(parents=True, exist_ok=False)
    (root / 'hardware_before.txt').write_text(before)
    import torch
    import vllm_fl
    from vllm.platforms import current_platform
    assert current_platform.vendor_name == 'iluvatar'
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.set_num_threads(8)
    base = Path(__file__).resolve().parent
    writer_path = base.parent / 'tianshu_soft8_mixed_planes_20261019/writer.py'
    gate = json.loads((base / 'CPU_GATE_GITHUB_32g.json').read_text())
    assert gate['complete'] and not gate['GPU_executed'] and not gate['adoption_allowed']
    for name in ('codec.py', 'check_cpu.py'):
        matches = [value for path, value in gate['frozen_sources'].items() if path.endswith('/' + name)]
        assert len(matches) == 1 and hashlib.sha256((base / name).read_bytes()).hexdigest() == matches[0]
    prior = codec._prior_codec()
    sources = [Path(__file__), Path(codec.__file__), writer_path, args.reader_source,
               Path(prior.__file__), Path(prior.legacy.__file__), base / 'CPU_GATE_GITHUB_32g.json']
    frozen = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for path in sources:
        copied = root / 'source_snapshot' / path.resolve().relative_to('/')
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(path.read_bytes())
    result = dict(complete=False, phase='GPU codec and writer checks', runner_pid=os.getpid(),
                  started_utc=datetime.now(timezone.utc).isoformat(), stop_at_utc=deadline.isoformat(),
                  frozen_sources=frozen, cases=[], CPU_gate_runner='GitHub Ubuntu 24.04',
                  actual_GPU=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                  attention_forward_executed=False, physical_page_spec_connected=False,
                  service_executed=False, official_benchmark=False, adoption_allowed=False,
                  Graph_scope='KV writer only; Q/block-table/attention Graph not tested')
    def save():
        temporary = root / 'result.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(root / 'result.json')
    save()
    def expected_storage(raw, k, v, key, value, slots, denominator):
        expected = bytearray([17]) * raw.numel()
        keys = key.cpu().view(torch.int16).tolist()
        values = value.cpu().float().tolist()
        for row, slot in enumerate(slots.cpu().tolist()):
            if slot < 0:
                continue
            block, token = divmod(slot, 16)
            assert block < k.shape[0]
            for head in range(2):
                for dim in range(128):
                    kp = k.storage_offset() + block * k.stride(0) + token * k.stride(1) + head * k.stride(2) + dim
                    vp = v.storage_offset() + block * v.stride(0) + token * v.stride(1) + head * v.stride(2) + dim
                    expected[kp] = prior.legacy.encode_bits(keys[row][head][dim] & 65535)
                    expected[vp] = codec.encode_scalar(values[row][head][dim], denominator)
        return expected
    try:
        with torch.inference_mode():
            bits = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
            all_bf16 = bits.view(torch.bfloat16)
            all_codes = torch.arange(256, device='cuda', dtype=torch.int32).to(torch.uint8)
            key_expected = bytes(prior.legacy.encode_bits(i) for i in range(65536))
            for denominator in codec.DENOMINATORS:
                reader, writer, proof = codec.load_modules(args.reader_source, writer_path,
                                                          root / f'generated_d{denominator}', denominator)
                case = dict(denominator=denominator, source_proof=proof, writer_fixtures=[])
                result['cases'].append(case)
                save()
                encoded = torch.full((65536 + 17,), 165, device='cuda', dtype=torch.uint8)
                compiled = writer._linear_v_encode_all[(256,)](all_bf16, encoded, 65536,
                                                              num_warps=1, num_stages=1)
                torch.cuda.synchronize()
                expected = bytes(codec.encode_scalar(codec.bf16_scalar(i), denominator) for i in range(65536))
                case['all65536_GPU_V_encodings_sha256'] = equal_bytes(encoded[:65536], expected, 'all BF16 V encodings')
                assert bool((encoded[65536:] == 165).all())
                (root / f'GPU_V_encoded_d{denominator}.bin').write_bytes(bytes(encoded[:65536].cpu().tolist()))
                case['V_encoder_metadata'] = metadata(compiled)
                decoded = torch.empty(256, device='cuda', dtype=torch.bfloat16)
                reader._linear_v_all_codes[(1,)](all_codes, decoded, num_warps=1, num_stages=1)
                assert decoded.cpu().float().tolist() == [codec.decode_scalar(i, denominator) for i in range(256)]
                key_encoded = torch.empty(65536, device='cuda', dtype=torch.uint8)
                writer.encode_all[(256,)](all_bf16, key_encoded, 65536, num_warps=1, num_stages=1)
                case['all65536_GPU_K_encodings_sha256'] = equal_bytes(key_encoded, key_expected, 'unchanged K encodings')
                reader._codec_all_codes[(1,)](all_codes, decoded, num_warps=1, num_stages=1)
                actual_k_bits = [value & 65535 for value in decoded.cpu().view(torch.int16).tolist()]
                expected_k_bits = [(((i & 127) << 4) + 14848 if (i & 127) >= 8 else 0) | ((i & 128) << 8) for i in range(256)]
                assert actual_k_bits == expected_k_bits
                boundaries = [(integer + .5 + delta) / denominator for integer in range(-130, 131)
                              for delta in (-1 / 1024, 0, 1 / 1024)]
                boundary_tensor = torch.tensor(boundaries, device='cuda', dtype=torch.float32)
                boundary_encoded = torch.full((len(boundaries) + 17,), 165, device='cuda', dtype=torch.uint8)
                writer._linear_v_encode_all[((len(boundaries) + 255) // 256,)](
                    boundary_tensor, boundary_encoded, len(boundaries), num_warps=1, num_stages=1)
                equal_bytes(boundary_encoded[:len(boundaries)],
                            [codec.encode_scalar(x, denominator) for x in boundary_tensor.cpu().tolist()], 'FP32 ties/saturation boundaries')
                assert bool((boundary_encoded[len(boundaries):] == 165).all())
                case.update(V_codepoints=256, K_codepoints_and_signed_zero_bits=256,
                            FP32_rounding_boundaries=len(boundaries), encoder_tail_guard=True)
                for packed in (False, True):
                    torch.manual_seed(2026102000 + denominator + int(packed))
                    page_stride = 8448 if packed else 8192
                    offset = 128 if packed else 0
                    raw = torch.full((page_stride * 5 + 256,), 17, device='cuda', dtype=torch.uint8)
                    full = torch.as_strided(raw, (5, 2, 16, 2, 128),
                                            (page_stride, 4096, 256, 128, 1), storage_offset=offset)
                    kv = full[::2]
                    k, v = kv.unbind(1)
                    key = torch.randn((6, 4, 128), device='cuda', dtype=torch.bfloat16)[:, :2]
                    value = torch.linspace(-20, 20, 6 * 4 * 128, device='cuda').reshape(6, 4, 128).to(torch.bfloat16)[:, :2]
                    slots = torch.tensor([0, 15, 16, 31, 47, -1], device='cuda', dtype=torch.int64)
                    fixture = dict(packed=packed, raw_bytes=raw.numel(), page_stride_bytes=page_stride,
                                   cache_strides=list(kv.stride()), raw_offset=offset, input_stride=list(key.stride()),
                                   physical_page_target_bytes=8192, phases=[])
                    case['writer_fixtures'].append(fixture)
                    writer.write(key, value, k, v, slots)
                    torch.cuda.synchronize()
                    equal_bytes(raw, expected_storage(raw, k, v, key, value, slots, denominator), 'writer eager full-storage guards')
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        writer.write(key, value, k, v, slots)
                    patterns = ([0, 15, 16, 31, 47, -1], [46, -1, 1, 30, 17, 0], [-1, 32, 2, 18, 45, 14])
                    for phase, pattern in enumerate(patterns):
                        if phase:
                            key.add_(.375)
                            value.neg_().add_(.25)
                            slots.copy_(torch.tensor(pattern, device='cuda', dtype=torch.int64))
                        raw.fill_(17)
                        expected_raw = expected_storage(raw, k, v, key, value, slots, denominator)
                        graph.replay()
                        torch.cuda.synchronize()
                        digest = equal_bytes(raw, expected_raw, 'writer Graph changed inputs/slots and complete storage')
                        filename = f'writer_d{denominator}_packed{int(packed)}_phase{phase}.pt'
                        torch.save(dict(raw=raw.cpu(), key=key.cpu(), value=value.cpu(), slots=slots.cpu(),
                                        expected_raw=bytes(expected_raw), cache_strides=kv.stride(),
                                        k_offset=k.storage_offset(), v_offset=v.storage_offset()), root / filename)
                        fixture['phases'].append(dict(phase=phase, slots=pattern, complete_storage_matches=True,
                                                      negative_slot_and_all_guards_preserved=True,
                                                      actual_raw_sha256=digest, raw_fixture=filename))
                    del graph
                    save()
                case['complete'] = True
                save()
            torch.cuda.synchronize()
            result.update(complete=True, phase='complete', ended_utc=datetime.now(timezone.utc).isoformat(),
                          GPU_codecs_match_CPU_contract=True, GPU_K_bits_preserved=True,
                          GPU_writer_Graph_changed_inputs_passed=True)
            save()
    except BaseException as error:
        result.update(phase='failed', error=repr(error), ended_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise
    finally:
        (root / 'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'], text=True))
    print(json.dumps(dict(complete=result['complete'], output=str(root), scales=len(result['cases']),
                         scope=result['phase'], attention_forward_executed=False, adoption_allowed=False)))


if __name__ == '__main__':
    main()