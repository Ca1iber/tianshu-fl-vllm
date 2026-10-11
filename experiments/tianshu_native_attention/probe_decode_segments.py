# SPDX-License-Identifier: Apache-2.0
"""Actual vLLM 3D Decode sweep; no service or installed source changes."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import statistics
import time

import torch
import vllm_fl  # activate the installed platform compatibility initialization
import vllm.v1.attention.ops.triton_unified_attention as native


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def check(value, expected):
    a, b = value.float(), expected.float()
    diff = (a - b).abs()
    bad = diff > .02 + .01 * b.abs()
    return {'passed': bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not bad.any()),
            'max_abs': diff.max().item(), 'mean_abs': diff.mean().item(),
            'failed_elements': bad.sum().item(), 'atol': .02, 'rtol': .01}


def reference(q, k, v, table, lengths):
    out = []
    for i, length in enumerate(lengths):
        pages = table[i, :((length + 15) // 16)].long()
        keys = k[pages].reshape(-1, 2, 128)[:length].float().repeat_interleave(8, dim=1)
        values = v[pages].reshape(-1, 2, 128)[:length].float().repeat_interleave(8, dim=1)
        query = q[i].float()[:, None, :]
        scores = torch.bmm(query, keys.permute(1, 2, 0)) * 128**-.5
        out.append(torch.bmm(scores.softmax(-1), values.permute(1, 0, 2))[:, 0])
    return torch.stack(out)


def run_case(batch, length, ragged, strided, seed):
    torch.manual_seed(seed)
    if length == 0:
        lengths = [1, 15, 16, 17, 127, 257, 1023, 4095]
        batch = len(lengths)
    else:
        lengths = [length - (i * 37 % 257) if ragged else length for i in range(batch)]
    width = (max(lengths) + 15) // 16
    pages = batch * width
    table = torch.randperm(pages, device='cuda').int().reshape(batch, width)
    q = torch.randn(batch, 20 if strided else 16, 128, device='cuda', dtype=torch.bfloat16)[:, :16]
    k = torch.randn(pages * (2 if strided else 1), 16, 2, 128, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    if strided:
        k, v = k[::2], v[::2]
    cu = torch.arange(batch + 1, device='cuda', dtype=torch.int32)
    used = torch.tensor(lengths, device='cuda', dtype=torch.int32)
    expected = reference(q, k, v, table, lengths)
    # Independent CPU FP32 calculation on three sequences and both KV groups.
    cpu_checks = []
    for seq in sorted({0, batch // 2, batch - 1}):
        ids = table[seq, :(lengths[seq] + 15)//16].cpu().long()
        keys = k.detach().cpu()[ids].reshape(-1, 2, 128)[:lengths[seq]].float()
        values = v.detach().cpu()[ids].reshape(-1, 2, 128)[:lengths[seq]].float()
        for head in (0, 7, 8, 15):
            query = q[seq, head].cpu().float()
            scores = (keys[:, head // 8] @ query) * 128**-.5
            cpu = scores.softmax(0) @ values[:, head // 8]
            cpu_checks.append(check(expected[seq, head].cpu(), cpu))
    graphs, outputs, buffers, variants = {}, {}, {}, []
    for segments in (4, 8, 16, 32, 64):
        output = torch.empty_like(q)
        partial = torch.empty((batch, 16, segments, 128), device='cuda', dtype=torch.float32)
        maximum = torch.empty((batch, 16, segments), device='cuda', dtype=torch.float32)
        expsum = torch.empty_like(maximum)
        def invoke():
            native.unified_attention(q=q, k=k, v=v, out=output, cu_seqlens_q=cu,
                max_seqlen_q=1, seqused_k=used, max_seqlen_k=max(lengths),
                softmax_scale=128**-.5, causal=True, window_size=(-1,-1),
                block_table=table, softcap=0., q_descale=None, k_descale=None, v_descale=None,
                seq_threshold_3D=64, num_par_softmax_segments=segments,
                softmax_segm_output=partial, softmax_segm_max=maximum,
                softmax_segm_expsum=expsum)
        for _ in range(3):
            invoke()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke()
        graph.replay()
        torch.cuda.synchronize()
        entry = {'segments': segments, 'vs_fp32': check(output, expected),
                 'buffer_bytes': sum(t.numel()*t.element_size() for t in (partial,maximum,expsum))}
        graphs[segments], outputs[segments], buffers[segments] = graph, output, (partial,maximum,expsum)
        variants.append(entry)
    timings = {s: [] for s in graphs}
    order = list(graphs)
    for trial in range(9):
        rotation = order[trial % len(order):] + order[:trial % len(order)]
        for s in rotation:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                graphs[s].replay()
            end.record(); end.synchronize()
            timings[s].append(start.elapsed_time(end) / 50)
    for entry in variants:
        s = entry['segments']
        entry['samples_ms'] = timings[s]
        entry['median_ms'] = statistics.median(timings[s])
        entry['vs_native_16'] = check(outputs[s], outputs[16])
        entry['speedup_over_native_16'] = statistics.median(timings[16]) / entry['median_ms']
    return {'batch': batch, 'kv_lengths': lengths, 'strided':strided, 'cpu_reference_checks':cpu_checks,
            'q_stride':list(q.stride()), 'kv_stride':list(k.stride()), 'variants':variants}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    signal.alarm(1800)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = {'official_benchmark':False, 'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'native_source':native.__file__, 'native_sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases':[]}
    cases = [(1,4096,False,False), (8,4096,False,False), (16,4096,False,False),
             (32,4096,False,False), (64,4096,False,False), (64,5120,False,False),
             (64,5120,True,True), (8,0,False,True)]
    with torch.inference_mode():
        for i,case in enumerate(cases):
            result = run_case(*case,2026101121+i)
            data['cases'].append(result)
            save(args.output/'result.json',data)
            print(json.dumps({'case':i, 'batch':result['batch'], 'variants':[
                {'segments':v['segments'],'ms':v['median_ms'],'speedup':v['speedup_over_native_16'],
                 'passed':v['vs_fp32']['passed']} for v in result['variants']]}),flush=True)
            torch.cuda.empty_cache()
    data['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    save(args.output/'result.json',data)


if __name__=='__main__':
    main()
