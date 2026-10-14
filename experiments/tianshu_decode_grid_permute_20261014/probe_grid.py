# SPDX-License-Identifier: Apache-2.0
"""Decode CTA-axis permutations on the .cv baseline, with changed-KV Graph checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import itertools
from pathlib import Path
import signal
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--b64-segments',type=int,choices=[8],default=8)
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
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "tianshu_decode_low_warps_20261014"))
    from launch_router import LaunchRouter
    asm_recorded=set()
    class PageRouter(LaunchRouter):
        def __init__(self,original,configuration,axes):
            outer=self
            self.axes=axes
            class GridKernel:
                def __getitem__(self,grid):
                    def launch(*a,**kw):
                        logical=(kw['query_ptr'].shape[0]//kw['BLOCK_Q']+kw['num_seqs'],grid[1],grid[2])
                        actual=logical if axes is None else permute_grid(logical,axes)
                        outer.actual_grid=list(actual)
                        outer.logical_grid=list(logical)
                        outer.actual_rows=kw['BLOCK_M']
                        outer.actual_block_q=kw['BLOCK_Q']
                        compiled=original[actual](*a,**kw)
                        outer.register_info=dict(n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),
                            requested_cache_modifier=kw.get('KV_CACHE_MODIFIER'))
                        tag='baseline_cv' if axes is None else ''.join(map(str,axes))
                        if case_id==4 and tag not in asm_recorded:
                            asm_recorded.add(tag)
                            out=root/'kernel_artifacts'/tag;out.mkdir(parents=True,exist_ok=True)
                            files={}
                            for name,value in compiled.asm.items():
                                assert name.replace('_','').isalnum()
                                payload=value.encode() if isinstance(value,str) else value if isinstance(value,bytes) else None
                                if payload is not None:
                                    (out/name).write_bytes(payload);files[name]=hashlib.sha256(payload).hexdigest()
                            (out/'manifest.json').write_text(json.dumps(dict(requested_cache_modifier=kw.get('KV_CACHE_MODIFIER'),files=files,register_info=outer.register_info,
                                logical_to_physical_axes=axes,logical_grid=list(logical),actual_grid=list(actual)),indent=2)+'\n')
                        return compiled
                    return launch
            super().__init__(GridKernel(),configuration)
        def __getitem__(self,grid):
            original_launch=super().__getitem__(grid)
            def launch(*a,**kw):
                result=original_launch(*a,**kw)
                self.metadata.update(actual_BLOCK_SIZE=kw['BLOCK_SIZE'],actual_BLOCK_M=self.actual_rows,
                                     actual_BLOCK_Q=self.actual_block_q,actual_grid=self.actual_grid,
                                     logical_grid=self.logical_grid,logical_to_physical_axes=self.axes,**self.register_info)
                return result
            return launch
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    native_source = Path(native.__file__)
    native_sha = hashlib.sha256(native_source.read_bytes()).hexdigest()
    assert native_sha == 'bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0'
    snapshot = root / 'source_snapshot'
    snapshot.mkdir()
    helper_source = Path(__file__).resolve().parents[1] / "tianshu_native_prefill_20261012/zero_initialization.py"
    for p in (native_source, Path(__file__), Path(__file__).resolve().parents[1] / 'tianshu_decode_low_warps_20261014/launch_router.py', helper_source):
        (snapshot / p.name).write_bytes(p.read_bytes())
    backend=Path('/usr/local/lib/python3.12/site-packages/triton/backends/iluvatar/compiler.py')
    compiler=Path('/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py')
    assert 'iluvatar-num-vgpr' in backend.read_text() and 'options.maxnreg' in backend.read_text()
    assert 'self.n_regs, self.n_spills' in compiler.read_text()
    (snapshot/'backend_compiler.py').write_bytes(backend.read_bytes())
    (snapshot/'compiled_kernel.py').write_bytes(compiler.read_bytes())
    provenance = {'baseline': "current guarded L0 KV16/w1; B24 fallbackKV32/w2; smallB w4", 'native_sha256': native_sha, 'grid_changed': True, 'grid_change_scope':'physical CTA axes only; full logical CTA set and BLOCK_Q2 preserved',
                  'kernel_body_changed': True, 'tile_reduction_order_may_change': False, 'producer_and_reducer_tile_matched': True, 'installed_sources_modified': False}
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
    assert args.zero_init, "Use the validated no-sink L0 kernel"
    sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'tianshu_decode_cache_policy_20261014'))
    from experimental_cache import load_kernel
    cached,cache_provenance=load_kernel(generated['generated_source'],root/'cache_policy_source')
    cache_source=Path(__file__).resolve().parents[1] / 'tianshu_decode_cache_policy_20261014/experimental_cache.py'
    (snapshot/cache_source.name).write_bytes(cache_source.read_bytes())
    provenance.update(cache_policy_generation=cache_provenance,cache_helper_sha256=hashlib.sha256(cache_source.read_bytes()).hexdigest())
    from grid_permute import load_kernel as load_permuted, permute_grid
    axes_list=[None]+list(itertools.permutations(range(3)))
    kernels=[cached];grid_generations=[]
    for axes in axes_list[1:]:
        kernel,record=load_permuted(cache_provenance['generated_source'],root/'grid_sources',axes)
        kernels.append(kernel);grid_generations.append(record)
    grid_helper=Path(__file__).with_name('grid_permute.py')
    (snapshot/grid_helper.name).write_bytes(grid_helper.read_bytes())
    provenance.update(grid_generations=grid_generations,grid_helper_sha256=hashlib.sha256(grid_helper.read_bytes()).hexdigest())
    data = {'complete': False, 'official_benchmark': False, 'adoption_allowed': False,
            'provenance': provenance, 'cases': [], 'started_utc': datetime.now(timezone.utc).isoformat(),
            'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'backend_source_sha256':{'backend_compiler.py':hashlib.sha256(backend.read_bytes()).hexdigest(),
                'compiled_kernel.py':hashlib.sha256(compiler.read_bytes()).hexdigest()}}

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

    scenarios=[(n,4096,False,False) for n in (8,16,24,32,64)]
    scenarios += [(64,5120,True,False),(64,65,True,True)]
    scenarios += [(8,n,False,False) for n in (1,15,16,17,31,32,33,63,64,65)]
    save()
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
                selected=batch in (8,16,32,40,48,56,64)
                plan=dict(TILE_SIZE=16 if selected else 32,num_warps=1 if selected else 2 if 8<=batch<=64 else 4,num_stages=1)
                plans=[dict(plan,KV_CACHE_MODIFIER='.cv') for _ in axes_list]
                routers=[PageRouter(kernel,p,axes) for kernel,p,axes in zip(kernels,plans,axes_list)]
                cu = torch.tensor([0] + qlens, device='cuda', dtype=torch.int32).cumsum(0).int()
                used = torch.tensor(lengths, device='cuda', dtype=torch.int32)
                segments = args.b64_segments if batch==64 else 64 if batch==1 else 4 if 16<=batch<=64 else 16
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
                    # Every candidate uses these exact same K/V/table allocations.
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
                            'configuration':plans[plan_id], 'query_row_tile':16, 'error':str(error)})
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
                # Same captured addresses, independent changed-query FP32/CPU references.
                q.neg_();k.neg_();v.mul_(.5)
                changed_expected=[]
                changed_cpu=[]
                qi=0
                for seq,qlen in enumerate(qlens):
                    if not qlen:continue
                    ids=table[seq,:(lengths[seq]+15)//16].long()
                    keys=k[ids].reshape(-1,2,128)[:lengths[seq]].float().repeat_interleave(8,1)
                    values=v[ids].reshape(-1,2,128)[:lengths[seq]].float().repeat_interleave(8,1)
                    scores=torch.bmm(q[qi].float()[:,None],keys.permute(1,2,0))[:,0]*128**-.5
                    changed_expected.append(torch.bmm(scores.softmax(-1)[:,None],values.permute(1,0,2))[:,0])
                    if qi in {0,queries//2,queries-1}:
                        kc=k[ids].cpu().reshape(-1,2,128)[:lengths[seq]].float()
                        vc=v[ids].cpu().reshape(-1,2,128)[:lengths[seq]].float()
                        for h in (0,7,8,15):
                            qc=q[qi,h].cpu().float()
                            changed_cpu.append((qi,h,((kc[:,h//8]@qc)*128**-.5).softmax(0)@vc[:,h//8]))
                    qi+=1
                changed_expected=torch.stack(changed_expected)
                for graph in graphs:graph.replay()
                torch.cuda.synchronize()
                changed_checks=[check(o[:queries],changed_expected) for o in outputs]
                changed_cpu_checks=[check(o[qi,h].cpu(),reference) for o in outputs for qi,h,reference in changed_cpu]
                changed_bitwise=[torch.equal(outputs[0],o) for o in outputs]
                if not all(v['passed'] for v in changed_checks+changed_cpu_checks):
                    data.setdefault('changed_input_failures',[]).append(dict(case_id=case_id,batch=batch,
                        fp32_checks=changed_checks,cpu_checks=changed_cpu_checks,
                        active_configurations=[plans[i] for i in active]))
                    save()
                    raise AssertionError('Changed query/K/V Graph verification failed')
                q.neg_();k.neg_();v.mul_(2)
                for graph in graphs:graph.replay()
                torch.cuda.synchronize()
                assert all(check(o[:queries],expected)['passed'] for o in outputs)
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
                          'strided':strided, 'segments':segments, 'logical_KV_identical':True, 'query_row_tiles':[16]*len(graphs), 'baseline_configuration':plans[0],
                          'original_vs_fp32':check(outputs[0][:queries], expected),
                          'changed_graph_fp32_checks':changed_checks,'changed_graph_cpu_checks':changed_cpu_checks,
                          'changed_graph_bitwise_equal':changed_bitwise,
                          'cpu_reference_checks':cpu_checks,
                          'baseline_ms':medians[0], 'baseline_tail_preserved':tails[0],
                          'baseline_launch_metadata':routers[0].metadata,
                          'variants':[]}
                for i in range(1,len(graphs)):
                    plan_id=active[i]
                    result['variants'].append({
                        'configuration':plans[plan_id], 'logical_to_physical_axes':axes_list[plan_id],
                        'query_row_tile':16, 'candidate_ms':medians[i],
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
                          and all(c['changed_graph_bitwise_equal'])
                          and all(x['passed'] for x in c['cpu_reference_checks']) for c in data['cases'])
        valid=[]
        for axes in axes_list[1:]:
            ok=all(any(v['logical_to_physical_axes']==axes
                and v['candidate_vs_fp32']['passed'] and v['candidate_vs_native']['passed']
                and v['candidate_bitwise_equal'] and v['tails_preserved']
                and v['launch_metadata']['requested_cache_modifier']=='.cv'
                for v in c['variants']) for c in data['cases'])
            if ok:valid.append(axes)
        data.update(complete=True,numerical_checks_passed=controls_ok and len(valid)==6,
                    valid_axis_permutations=valid,guarded_decode_baseline_preserved=True,b64_segments=args.b64_segments,
                    producer_and_reducer_tile_matched=True,changed_Graph_query_and_KV=True,
                    scope='CTA-axis permutation microprobe on .cv; no cache hierarchy/occupancy/model/quality claim',
                    finished_utc=datetime.now(timezone.utc).isoformat())
    finally:
        native.kernel_unified_attention = original
        native.reduce_segments = original_reduce
        native.kernel_unified_attention = installed_kernel
        save()


if __name__ == '__main__':
    main()
