# SPDX-License-Identifier: Apache-2.0
"""Native Decode KV tiles with a matched producer/reducer launch contract."""
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
    parser.add_argument('--short-boundaries',action='store_true')
    parser.add_argument('--zero-init',action='store_true')
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
    from launch_router import LaunchRouter
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    native_source = Path(native.__file__)
    native_sha = hashlib.sha256(native_source.read_bytes()).hexdigest()
    assert native_sha == 'bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0'
    snapshot = root / 'source_snapshot'
    snapshot.mkdir()
    for p in (native_source, Path(__file__), Path(__file__).with_name('launch_router.py')):
        (snapshot / p.name).write_bytes(p.read_bytes())
    provenance = {'native_sha256': native_sha, 'grid_changed': False,
                  'kernel_body_changed': False, 'tile_reduction_order_may_change': True, 'producer_and_reducer_tile_matched': True, 'installed_sources_modified': False}
    original_reduce = native.reduce_segments
    installed_kernel = native.kernel_unified_attention
    original = installed_kernel
    if args.zero_init:
        import importlib.util
        helper = Path(__file__).resolve().parents[1] / 'tianshu_native_prefill_20261012/zero_initialization.py'
        spec = importlib.util.spec_from_file_location('_tianshu_decode_zero_generator', helper)
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        original, generated = generator.load_kernel(native, root / 'zero_initialization')
        provenance.update(zero_initialization=generated, kernel_body_changed=True)
        (snapshot / 'zero_initialization.py').write_bytes(helper.read_bytes())
    plans = [{}, dict(TILE_SIZE=16, num_warps=4, num_stages=1),
             dict(TILE_SIZE=32, num_warps=4, num_stages=1),
             dict(TILE_SIZE=32, num_warps=4, num_stages=2)]
    routers = [LaunchRouter(original, plan) for plan in plans]
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

    scenarios = [(1,4096,False,False),(8,4096,False,False),(16,4096,False,False),
                 (32,4096,False,False),(64,4096,False,False),(64,5120,True,False),
                 (16,16384,True,False),(2,32768,True,False)]
    if args.short_boundaries:
        scenarios=[(1,n,False,False) for n in (1,7,13,15,16,17,31,32,33,63,64,65)]
        scenarios += [(64,33,True,True),(64,65,True,True)]
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
                cache = torch.randn(pages * (2 if strided else 1), 2, 16, 2, 128,
                                    device='cuda', dtype=q.dtype)
                k, v = cache.unbind(1)
                if strided:
                    k, v = k[::2], v[::2]
                cu = torch.tensor([0] + qlens, device='cuda', dtype=torch.int32).cumsum(0).int()
                used = torch.tensor(lengths, device='cuda', dtype=torch.int32)
                segments = 64 if batch == 1 else 4 if batch >= 16 else 16
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
                active = []
                for plan_id, selected in enumerate(routers):
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
                    try:
                        for _ in range(3):
                            invoke()
                    except Exception as error:
                        if plan_id==0 or type(error).__name__ not in (
                            'OutOfResources','CompilationError','UnsupportedLanguageConstruct',
                            'CompileTimeAssertionFailure'):
                            raise
                        data.setdefault('compile_errors',[]).append({
                            'batch':batch,'kv_max':max(lengths),'strided':strided,
                            'configuration':plans[plan_id],'error':str(error)})
                        save()
                        continue
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        invoke()
                    graph.replay()
                    torch.cuda.synchronize()
                    graphs.append(graph)
                    active.append(plan_id)
                    outputs.append(output)
                    buffers.append((partial, maximum, expsum))
                    tail = bool((output[queries:] == 17.).all())
                    for t in (partial, maximum, expsum):
                        count = queries * 16 * segments * (128 if t is partial else 1)
                        tail = tail and bool((t.flatten()[count:] == 17.).all())
                    tails.append(tail)
                measurements = [[] for _ in graphs]
                for repetition in range(9):
                    order = list(range(len(graphs)))
                    offset = repetition % len(order)
                    for index in order[offset:] + order[:offset]:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(50):
                            graphs[index].replay()
                        end.record()
                        end.synchronize()
                        measurements[index].append(start.elapsed_time(end) / 50)
                medians = list(map(statistics.median, measurements))
                result = {'batch':batch, 'query_lengths':qlens, 'kv_lengths':lengths,
                          'strided':strided, 'segments':segments,
                          'original_vs_fp32':check(outputs[0][:queries], expected),
                          'cpu_reference_checks':cpu_checks,
                          'baseline_ms':medians[0], 'baseline_tail_preserved':tails[0],
                          'baseline_launch_metadata':routers[0].metadata,
                          'variants':[]}
                for i in range(1,len(graphs)):
                    plan_id=active[i]
                    result['variants'].append({
                        'configuration':plans[plan_id], 'candidate_ms':medians[i],
                        'candidate_vs_fp32':check(outputs[i][:queries], expected),
                        'candidate_vs_native':check(outputs[i][:queries],outputs[0][:queries]),
                        'candidate_bitwise_equal':torch.equal(outputs[0],outputs[i]),
                        'tails_preserved':tails[i], 'raw_ms':measurements[i],
                        'launch_metadata':routers[plan_id].metadata,
                        'time_change_percent':100*(medians[i]/medians[0]-1)})
                result['baseline_raw_ms'] = measurements[0]
                data['cases'].append(result)
                save()
                print(json.dumps(result), flush=True)
                native.kernel_unified_attention = original
                del graphs, outputs, buffers
                torch.cuda.empty_cache()
        controls_ok = all(c['original_vs_fp32']['passed'] and c['baseline_tail_preserved']
                          and all(x['passed'] for x in c['cpu_reference_checks']) for c in data['cases'])
        valid = [plan for plan in plans[1:] if all(
            any(v['configuration'] == plan and v['candidate_vs_fp32']['passed']
                and v['candidate_vs_native']['passed'] and v['tails_preserved']
                for v in c['variants']) for c in data['cases'])]
        data.update(complete=True, configurations=plans, numerical_checks_passed=controls_ok,
                    valid_configurations=valid, producer_and_reducer_tile_matched=True,
                    finished_utc=datetime.now(timezone.utc).isoformat())
    finally:
        native.kernel_unified_attention = original
        native.reduce_segments = original_reduce
        native.kernel_unified_attention = installed_kernel
        save()


if __name__ == '__main__':
    main()
