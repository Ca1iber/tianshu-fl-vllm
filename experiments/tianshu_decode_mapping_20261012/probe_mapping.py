# SPDX-License-Identifier: Apache-2.0
"""Direct Decode mapping versus unchanged native attention and CPU FP32."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--small-kv-only', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'result.json').exists()
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 0
    signal.alarm(min(1800, remaining))
    hardware = subprocess.check_output(['ixsmi'], text=True)
    assert 'No running processes found' in hardware
    (root / 'hardware_before.txt').write_text(hardware)
    import torch
    import vllm_fl
    import vllm.v1.attention.ops.triton_unified_attention as native
    from experimental_mapping import KernelRouter, load_kernel
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    candidate, provenance = load_kernel(native, root / 'source_snapshot')
    original = native.kernel_unified_attention
    router = KernelRouter(original, candidate)
    data = {'complete': False, 'official_benchmark': False, 'adoption_allowed': False,
            'provenance': provenance, 'cases': [], 'started_utc': datetime.now(timezone.utc).isoformat(),
            'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        temp = root / 'result.tmp'
        temp.write_text(json.dumps(data, indent=2) + '\n')
        temp.replace(root / 'result.json')

    def check(a, b):
        a, b = a.float(), b.float()
        diff = (a - b).abs()
        bad = diff > .02 + .01 * b.abs()
        return {'passed': bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not bad.any()),
                'max_abs': diff.max().item(), 'failed_elements': bad.sum().item(), 'atol': .02, 'rtol': .01}

    scenarios = [(1, 4096, False, False), (2, 4096, False, False), (8, 4096, False, False),
                 (16, 4096, False, False), (32, 4096, False, False), (64, 4096, False, False),
                 (8, 5120, True, False), (64, 5120, True, False),
                 (8, 257, True, True), (64, 4096, True, True)]
    if args.small_kv_only:
        scenarios = [(1,length,False,False) for length in (13,14,15,16,21,23,30)]
    try:
        with torch.inference_mode():
            for case_id, (batch, length, strided, padded) in enumerate(scenarios):
                torch.manual_seed(2026101500 + case_id)
                qlens = [int(i % 3 != 1) if padded else 1 for i in range(batch)]
                lengths = [max(1, length - i * 37 % min(257, length)) for i in range(batch)]
                queries = sum(qlens)
                width = (max(lengths) + 15) // 16
                pages = batch * width
                table = torch.randperm(pages, device='cuda').int().reshape(batch, width)
                q = torch.randn(queries, 20 if strided else 16, 128, device='cuda', dtype=torch.bfloat16)[:, :16]
                k = torch.randn(pages * (2 if strided else 1), 16, 2, 128, device='cuda', dtype=q.dtype)
                v = torch.randn_like(k)
                if strided:
                    k, v = k[::2], v[::2]
                cu = torch.tensor([0] + qlens, device='cuda', dtype=torch.int32).cumsum(0).int()
                used = torch.tensor(lengths, device='cuda', dtype=torch.int32)
                segments = 64 if batch == 1 else 4 if batch >= 16 else 16
                if args.small_kv_only:
                    segments = 16  # Actual non-adaptive eager target path.
                expected = []
                cpu_checks = []
                query_index = 0
                for seq, qlen in enumerate(qlens):
                    if not qlen:
                        continue
                    ids = table[seq, :(lengths[seq] + 15) // 16].long()
                    keys = k[ids].reshape(-1, 2, 128)[:lengths[seq]].float().repeat_interleave(8, 1)
                    values = v[ids].reshape(-1, 2, 128)[:lengths[seq]].float().repeat_interleave(8, 1)
                    scores = torch.bmm(q[query_index].float()[:, None], keys.permute(1, 2, 0))[:, 0] * 128 ** -.5
                    expected.append(torch.bmm(scores.softmax(-1)[:, None], values.permute(1, 0, 2))[:, 0])
                    if query_index in {0, queries // 2, queries - 1}:
                        kc = k[ids].cpu().reshape(-1, 2, 128)[:lengths[seq]].float()
                        vc = v[ids].cpu().reshape(-1, 2, 128)[:lengths[seq]].float()
                        for head in (0, 7, 8, 15):
                            qc = q[query_index, head].cpu().float()
                            cpu = ((kc[:, head // 8] @ qc) * 128 ** -.5).softmax(0) @ vc[:, head // 8]
                            cpu_checks.append(check(expected[-1][head].cpu(), cpu))
                    query_index += 1
                expected = torch.stack(expected)
                graphs, outputs, buffers, tails = [], [], [], []
                for selected in (original, router):
                    native.kernel_unified_attention = selected
                    output = torch.full((queries + 3, 16, 128), 17., device='cuda', dtype=q.dtype)
                    partial = torch.full((queries, 16, 64, 128), 17., device='cuda', dtype=torch.float32)
                    maximum = torch.full((queries, 16, 64), 17., device='cuda', dtype=torch.float32)
                    expsum = torch.full_like(maximum, 17.)

                    def invoke():
                        native.unified_attention(q=q, k=k, v=v, out=output, cu_seqlens_q=cu,
                            max_seqlen_q=1, seqused_k=used, max_seqlen_k=max(lengths), softmax_scale=128 ** -.5,
                            causal=True, window_size=(-1, -1), block_table=table, softcap=0.,
                            q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=64,
                            num_par_softmax_segments=segments, softmax_segm_output=partial,
                            softmax_segm_max=maximum, softmax_segm_expsum=expsum)
                    for _ in range(3):
                        invoke()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        invoke()
                    graph.replay()
                    torch.cuda.synchronize()
                    graphs.append(graph)
                    outputs.append(output)
                    buffers.append((partial, maximum, expsum))
                    tail = bool((output[queries:] == 17.).all())
                    for t in (partial, maximum, expsum):
                        count = queries * 16 * segments * (128 if t is partial else 1)
                        tail = tail and bool((t.flatten()[count:] == 17.).all())
                    tails.append(tail)
                measurements = [[], []]
                for repetition in range(9):
                    for index in ((0, 1) if repetition % 2 == 0 else (1, 0)):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(50):
                            graphs[index].replay()
                        end.record()
                        end.synchronize()
                        measurements[index].append(start.elapsed_time(end) / 50)
                before, after = map(statistics.median, measurements)
                result = {'batch': batch, 'query_lengths': qlens, 'kv_lengths': lengths, 'strided': strided,
                          'segments': segments, 'original_vs_fp32': check(outputs[0][:queries], expected),
                          'candidate_vs_fp32': check(outputs[1][:queries], expected),
                          'candidate_vs_native': check(outputs[1][:queries], outputs[0][:queries]),
                          'candidate_bitwise_equal': torch.equal(outputs[0], outputs[1]),
                          'cpu_reference_checks': cpu_checks, 'tails_preserved': tails,
                          'baseline_ms': before, 'candidate_ms': after, 'raw_ms': measurements,
                          'change_percent': 100 * (after / before - 1)}
                data['cases'].append(result)
                save()
                print(json.dumps(result), flush=True)
                native.kernel_unified_attention = original
                del graphs, outputs, buffers
                torch.cuda.empty_cache()
        data.update(complete=True, routing_grids=router.routes, finished_utc=datetime.now(timezone.utc).isoformat())
    finally:
        native.kernel_unified_attention = original
        save()


if __name__ == '__main__':
    main()
