# SPDX-License-Identifier: Apache-2.0
"""Direct compressed Q2 component; real dense-gather baseline, no model score."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import traceback
from types import SimpleNamespace
import source
import launch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stop-at-utc', required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.stop_at_utc)
    assert deadline.utcoffset() is not None
    seconds = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert seconds > 0
    signal.alarm(min(seconds, 1800))
    hardware = subprocess.check_output(['ixsmi'], text=True)
    assert 'Iluvatar BI-V150' in hardware and 'No running processes found' in hardware
    base = Path(__file__).resolve().parent
    gate = json.loads((base/'CPU_GATE_32g.json').read_text())
    assert gate['complete'] and gate['CPU_only']
    for path, digest in gate['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    root = args.output.resolve()
    assert root.is_relative_to(Path('/workspace/benchmark-results'))
    root.mkdir(parents=True, exist_ok=False)
    (root/'hardware_before.txt').write_text(hardware)
    generated, proof = source.generate()
    path = root/'generated_q2_reader.py'
    path.write_text(generated)
    result = dict(complete=False, phase='preparing', runner_pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(), stop_at_utc=args.stop_at_utc,
        proof=proof, frozen_sources=gate['sources'], cases=[], GPU_component_only=True,
        model_executed=False, service_executed=False, official_benchmark=False, adoption_allowed=False)
    def save():
        pending = root/'result.tmp'
        pending.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        pending.replace(root/'result.json')
    save()
    try:
        import torch
        import vllm_fl
        from vllm.platforms import current_platform
        from vllm_fl.worker import iluvatar_hybrid_attention as hybrid
        from vllm.v1.attention.ops import triton_unified_attention as native
        assert current_platform.vendor_name == 'iluvatar' and torch.cuda.device_count() == 1
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        parent = base.parent/'tianshu_k8v16_pipeline_20261019'
        sys.path.insert(0, str(parent))
        from bounded_prefill import BoundedDensePrefill
        planes = load('_q2_value_planes', parent/'value_planes.py')
        dense = BoundedDensePrefill()
        dense.reserve()
        models = []
        for tile, warps in ((16, 1), (32, 2), (64, 4)):
            reader = load(f'_q2_direct_T{tile}', path)
            producer = launch.attach(reader, tile, warps)
            models.append(dict(label=f'direct_T{tile}_W{warps}', reader=reader, producer=producer))
        runtime_sources = dict(gate['sources'])
        from flag_gems import flash_attn_varlen_func
        import inspect
        for obj in (native, hybrid, flash_attn_varlen_func):
            actual = Path(inspect.getsourcefile(obj)).resolve()
            runtime_sources[str(actual)] = hashlib.sha256(actual.read_bytes()).hexdigest()
        runtime_sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        result.update(runtime_sources=runtime_sources, device=torch.cuda.get_device_name(),
            torch_version=torch.__version__, dense_arena_reservation=dense.reservation,
            timing='hot CUDA Graph component: 9 alternating trials x 20 replays; dense baseline includes actual K8->BF16 gather',
            phase='testing')
        for filename in runtime_sources:
            actual = Path(filename)
            copied = root/'source_snapshot'/actual.relative_to('/')
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(actual.read_bytes())
        def encode(x):
            bits = x.view(torch.int16).to(torch.int32)
            mag = bits & 32767
            rounded = ((mag+7+((mag>>4)&1))>>4)-928
            return (torch.where(rounded>=8, rounded.clamp(max=127), 0)|((bits&32768)>>8)).to(torch.uint8)
        def decode(x):
            bits=x.to(torch.int32); mag=bits&127
            return (torch.where(mag>=8,(mag<<4)+14848,0)|((bits&128)<<8)).to(torch.int16).view(torch.bfloat16)
        def compare(actual, expected):
            actual, expected = actual.float(), expected.float()
            delta = (actual-expected).abs()
            failed = delta > .002 + .01*expected.abs()
            return dict(passed=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all() and not failed.any()),
                max_abs=delta.max().item(), failed_elements=failed.sum().item(), atol=.002, rtol=.01)
        cases = [(f'q2_N{n}_K{length}', [2]*n, [length]*n, False)
            for n in (1, 8, 32, 64) for length in (4096, 5120)]
        cases += [('q1_N8_K4096',[1]*8,[4096]*8,False),
            ('mixed_N8_tiny_strided',[1,2,1,2,2,1,2,1],[1,2,3,5,17,33,65,127],True),
            ('mixed_N64_K4096_strided',[1 if i%3==0 else 2 for i in range(64)],
                [4096-(i%7)*131 for i in range(64)],True)]
        with torch.inference_mode():
            for index,(name,qlens,initial_lengths,strided) in enumerate(cases):
                torch.manual_seed(2026102100+index)
                n, tokens = len(qlens), sum(qlens)
                width=(max(initial_lengths)+15)//16
                pages=n*width
                q_storage=torch.randn(tokens+3,20 if strided else 16,128,device='cuda',dtype=torch.bfloat16)
                q=q_storage[:tokens,:16]
                logical_k=torch.randn(n,width*16,2,128,device='cuda',dtype=torch.bfloat16)
                logical_v=torch.randn_like(logical_k)
                raw=torch.full(((pages+3)*(2 if strided else 1)*12288,),17,device='cuda',dtype=torch.uint8)
                physical_pages=raw.numel()//12288
                physical=torch.as_strided(raw,(physical_pages,2,16,2,128),(12288,4096,256,128,1))
                physical=physical[::2] if strided else physical
                k,v=physical.unbind(1)
                v=planes.value_plane(v)
                table_storage=torch.zeros((n,width+3 if strided else width),device='cuda',dtype=torch.int32)
                table=table_storage[:,:width]
                used=torch.tensor(initial_lengths,device='cuda',dtype=torch.int32)
                cumulative=[0]
                for length in qlens:cumulative.append(cumulative[-1]+length)
                cu=torch.tensor(cumulative,device='cuda',dtype=torch.int32)
                scalar=torch.ones(1,device='cuda')
                segments=4 if n>24 else 16
                buffers=dict(softmax_segm_output=torch.full((tokens,16,segments,128),17.,device='cuda'),
                    softmax_segm_max=torch.full((tokens,16,segments),17.,device='cuda'),
                    softmax_segm_expsum=torch.full((tokens,16,segments),17.,device='cuda'))
                common=dict(q=q,k=k,v=v,cu_seqlens_q=cu,max_seqlen_q=max(qlens),seqused_k=used,
                    max_seqlen_k=width*16,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),
                    block_table=table,softcap=0.,q_descale=scalar,k_descale=scalar,v_descale=scalar,
                    seq_threshold_3D=64,num_par_softmax_segments=segments,**buffers)
                fixtures=[]
                for model in models:
                    storage=torch.full((tokens+3,20 if strided else 16,128),17.,device='cuda',dtype=torch.bfloat16)
                    out=storage[:tokens,:16]
                    kwargs=dict(common,out=out)
                    def invoke(reader=model['reader'],kwargs=kwargs):reader.unified_attention(**kwargs)
                    fixtures.append(dict(label=model['label'],out=out,storage=storage,invoke=invoke,graph=None))
                if all(x==2 for x in qlens):
                    storage=torch.full((tokens+3,16,128),17.,device='cuda',dtype=torch.bfloat16)
                    out=storage[:tokens]
                    kwargs=dict(common,out=out)
                    route=SimpleNamespace(num_decodes=0,prefill_cu=cu,max_prefill_query=2)
                    def baseline(kwargs=kwargs,route=route):
                        staged=dense(kwargs)
                        assert staged is not None
                        return hybrid.dispatch(native.unified_attention,route,staged)
                    fixtures.append(dict(label='dense_gather_BF16_hybrid',out=out,storage=storage,invoke=baseline,graph=None))
                phases=[]
                for phase in range(2):
                    lengths=initial_lengths if phase==0 else [max(qi,l-min(7,l-qi)) for qi,l in zip(qlens,initial_lengths)]
                    if phase:
                        q.neg_();logical_k.mul_(.5);logical_v.neg_()
                    used.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
                    table.copy_(torch.randperm(pages,device='cuda').int().reshape(n,width))
                    encoded=encode(logical_k)
                    decoded=decode(encoded)
                    ids=table.flatten().long()
                    k[ids]=encoded.reshape(pages,16,2,128)
                    v[ids]=logical_v.reshape(pages,16,2,128)
                    references=[]
                    for seq,(qi,length) in enumerate(zip(qlens,lengths)):
                        query=q[cumulative[seq]:cumulative[seq+1]].float().reshape(qi,2,8,128)
                        scores=torch.einsum('thgd,khd->thgk',query,decoded[seq].float())*(128**-.5)
                        positions=torch.arange(width*16,device='cuda')[None,:]
                        limits=(length-qi+torch.arange(qi,device='cuda'))[:,None]
                        scores.masked_fill_((positions>limits)[:,None,None,:],-float('inf'))
                        references.append(torch.einsum('thgk,khd->thgd',scores.softmax(-1),logical_v[seq].float()).reshape(qi,16,128))
                    expected=torch.cat(references)
                    checks={}
                    for fixture in fixtures:
                        fixture['invoke']();torch.cuda.synchronize()
                        checks[fixture['label']]=compare(fixture['out'],expected)
                    selected=sorted({0,n//2,n-1})
                    samples=[]
                    for seq in selected:
                        samples.append(dict(sequence=seq,query=q[cumulative[seq]:cumulative[seq+1]].cpu(),
                            K=decoded[seq,:lengths[seq]].cpu(),V=logical_v[seq,:lengths[seq]].cpu(),
                            length=lengths[seq],start=cumulative[seq]))
                    payload=dict(name=name,phase=phase,qlens=qlens,lengths=lengths,samples=samples,
                        expected=expected.cpu(),outputs={f['label']:f['out'].cpu() for f in fixtures})
                    torch.save(payload,root/f'{name}_phase{phase}.pt')
                    assert all(v['passed'] for v in checks.values()), (name,phase,checks)
                    cpu_checks=[]
                    for sample in samples:
                        qi=sample['query'].shape[0]
                        for local in range(qi):
                            valid=sample['length']-qi+local+1
                            for head in (0,7,8,15):
                                scores=(sample['K'][:valid,head//8].float()@sample['query'][local,head].float())*(128**-.5)
                                expected_cpu=scores.softmax(0)@sample['V'][:valid,head//8].float()
                                for fixture in fixtures:
                                    check=compare(fixture['out'][sample['start']+local,head].cpu(),expected_cpu)
                                    assert check['passed'], (name,phase,fixture['label'],sample['sequence'],local,head,check)
                                    cpu_checks.append(check)
                    if phase==0:
                        for fixture in fixtures:
                            fixture['invoke']();torch.cuda.synchronize()
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):fixture['invoke']()
                            fixture['graph']=graph
                            for _ in range(5):graph.replay()
                        torch.cuda.synchronize()
                    times={f['label']:[] for f in fixtures}
                    for trial in range(9):
                        order=fixtures if trial%2==0 else list(reversed(fixtures))
                        for fixture in order:
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(20):fixture['graph'].replay()
                            end.record();end.synchronize()
                            times[fixture['label']].append(start.elapsed_time(end)/20)
                    replay_checks={f['label']:compare(f['out'],expected) for f in fixtures}
                    payload['timed_outputs']={f['label']:f['out'].cpu() for f in fixtures}
                    torch.save(payload,root/f'{name}_phase{phase}.pt')
                    assert all(v['passed'] for v in replay_checks.values()), (name,phase,replay_checks)
                    for fixture in fixtures:
                        assert bool((fixture['storage'][tokens:]==17).all())
                        if strided:assert bool((fixture['storage'][:tokens,16:]==17).all())
                    row=dict(phase=phase,GPU_reference_checks=checks,CPU_checks=len(cpu_checks),CPU_checks_passed=True,
                        CUDA_graph_replay_checks=replay_checks,times_ms=times,
                        median_ms={key:statistics.median(values) for key,values in times.items()})
                    phases.append(row)
                    print(json.dumps(dict(case=name,phase=phase,median_ms=row['median_ms'])),flush=True)
                    result['cases'].append(dict(name=name,query_lengths=qlens,lengths=lengths,strided=strided,
                        scratch_query_token_capacity=tokens,segments=segments,**row))
                    save()
                del fixtures,common,buffers,logical_k,logical_v,q,q_storage,raw,k,v,physical,decoded,encoded,payload
                gc.collect();torch.cuda.empty_cache()
        result.update(complete=True,phase='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
            launches={model['label']:model['producer'].records for model in models},
            full_GPU_outputs_checked_against_FP32_causal_reference=True,
            CPU_independent_selected_heads_checked=True,
            changed_inputs_and_page_tables_replayed=True)
        save()
    except BaseException as exc:
        result.update(phase='failed',error=repr(exc),traceback=traceback.format_exc(),finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__=='__main__':main()
