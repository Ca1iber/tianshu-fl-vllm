# SPDX-License-Identifier: Apache-2.0
"""Check paired GateUp/SiLU against native GEMM, FP32 and CPU, then time graphs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import signal
import statistics
import subprocess
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--affine',action='store_true')
    args=parser.parse_args()
    root=args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    lease=Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    deadline=datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])
    remaining=int((deadline-datetime.now(timezone.utc)).total_seconds())
    assert remaining>0
    signal.alarm(min(1800, remaining))
    hardware=subprocess.check_output(['ixsmi'], text=True)
    assert 'No running processes found' in hardware
    (root/'hardware_before.txt').write_text(hardware)
    import torch
    import torch.nn.functional as F
    import vllm_fl
    import flag_gems
    from experimental_fusion import launch
    vendor=importlib.import_module('_iluvatar.ops.mm')
    vendor_source=Path(vendor.__file__)
    assert hashlib.sha256(vendor_source.read_bytes()).hexdigest()=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
    model_source=Path('/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/llama.py')
    assert hashlib.sha256(model_source.read_bytes()).hexdigest()=='1e1003454985bf86d3868a7755bcf5324676881548b3895809edf0b76698d384'
    assert json.loads(Path('/workspace/MiniCPM5-2B/config.json').read_text())['architectures']==['LlamaForCausalLM']
    source=Path(__file__).with_name('experimental_fusion.py')
    snapshots=root/'source_snapshot';snapshots.mkdir()
    for p in (source, Path(__file__), vendor_source, model_source):
        (snapshots/p.name).write_bytes(p.read_bytes())
    # Freeze a matching cached Inductor epilogue, rather than assuming eager
    # BF16 F.silu rounding describes the compiled model.
    body=None
    needle='def triton_poi_fused_mul_silu_slice_1'
    for p in Path('/root/.cache/vllm/torch_compile_cache').rglob('*.py'):
        if 'inductor' not in str(p):continue
        text=p.read_text(errors='replace')
        if needle in text:
            start=text.index(needle); candidate=text[start:text.find('\n\n', start)]
            if '12288*x1' in candidate and '6144' in candidate and 'tl.sigmoid' in candidate:
                assert 'tl.bfloat16' not in candidate
                body=candidate
                (snapshots/'compiled_activation_source.py').write_text(text)
                activation_source=str(p)
                break
    assert body, 'No matching compiled model epilogue found'
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    data=dict(complete=False, official_benchmark=False, adoption_allowed=False,
              installed_sources_modified=False, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              original_activation_source=activation_source, original_activation_body=body,
              projection_rounding='BF16 before FP32 SiLU/product; final BF16', cases=[],
              started_utc=datetime.now(timezone.utc).isoformat())

    def save():
        p=root/'result.tmp';p.write_text(json.dumps(data, indent=2)+'\n');p.replace(root/'result.json')

    case_atol=.1
    def check(a,b):
        a,b=a.float(),b.float();diff=(a-b).abs();bad=diff>case_atol+.02*b.abs()
        return dict(passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not bad.any()),
                    failed_elements=bad.sum().item(), max_abs=diff.max().item(), atol=case_atol, rtol=.02)

    def activation(projected, output):
        inter=projected.shape[1]//2
        # Explicit FP32 matches the actual cached compiled epilogue.
        gate=projected[:,:inter].float();up=projected[:,inter:].float()
        output.copy_(F.silu(gate)*up)
        return output

    compiled_activation=torch.compile(activation, fullgraph=True)
    plans=[dict(bm=64,bn=128,bk=32,warps=8,stages=1),
           dict(bm=64,bn=128,bk=32,warps=16,stages=2),
           dict(bm=64,bn=256,bk=32,warps=8,stages=1),
           dict(bm=32,bn=128,bk=32,warps=8,stages=1),
           dict(bm=32,bn=256,bk=32,warps=16,stages=1),
           dict(bm=64,bn=128,bk=64,warps=8,stages=1),
           dict(bm=64,bn=256,bk=64,warps=16,stages=1)]
    if args.affine:
        for plan in plans:plan['affine']=True
    scenarios=[(8,6144,2048,.25,.02),(16,6144,2048,.25,.02),
               (32,6144,2048,.25,.02),(64,6144,2048,.25,.02),
               (64,6144,2048,1.,1.),(17,6144,2048,.25,.02),
               (17,130,256,1.,1.),(1,130,256,.25,.02)]
    save()
    with torch.inference_mode():
        for case_id,(m,inter,k,xscale,wscale) in enumerate(scenarios):
            case_atol=.001 if xscale*wscale<.1 else .1
            torch.manual_seed(2026101700+case_id)
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*xscale
            weight=torch.randn(2*inter,k,device='cuda',dtype=x.dtype)*wscale
            reference_projection=(x.float()@weight.float().T).to(torch.bfloat16).float()
            gate,up=reference_projection[:,:inter],reference_projection[:,inter:]
            reference=(F.silu(gate)*up).to(x.dtype)
            cpu_checks=[]
            for row in sorted({0,m//2,m-1}):
                for column in sorted({0,inter//2,inter-1}):
                    wc=weight[[column,inter+column]].cpu().float()
                    pair=(wc@x[row].cpu().float()).to(torch.bfloat16).float()
                    expected=(F.silu(pair[0])*pair[1]).to(x.dtype)
                    cpu_checks.append(check(reference[row,column].cpu(),expected))
            projected=torch.empty(m,2*inter,device='cuda',dtype=x.dtype)
            storage=torch.full((m*inter+4096,),17.,device='cuda',dtype=x.dtype)
            baseline=storage[:m*inter].view(m,inter)

            def control():
                vendor._launch_mm(x,weight.T,projected,m,2*inter,k)
                compiled_activation(projected,baseline)

            for _ in range(3):control()
            graph_control=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_control):control()
            graph_control.replay();torch.cuda.synchronize()
            baseline_check=check(baseline,reference)
            baseline_tail=bool((storage[m*inter:]==17.).all())
            for config in plans:
                output_storage=torch.full((m*inter+4096,),17.,device='cuda',dtype=x.dtype)
                output=output_storage[:m*inter].view(m,inter)
                result=dict(m=m,intermediate=inter,k=k,x_scale=xscale,weight_scale=wscale,
                            configuration=config, cpu_reference_checks=cpu_checks,
                            baseline_vs_fp32=baseline_check, baseline_tail_preserved=baseline_tail)
                graph=None
                try:
                    for _ in range(3):launch(x,weight,output,config)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):launch(x,weight,output,config)
                    graph.replay();torch.cuda.synchronize()
                    result.update(candidate_vs_fp32=check(output,reference),
                                  candidate_vs_native=check(output,baseline),
                                  candidate_bitwise_equal=torch.equal(output,baseline),
                                  output_tail_preserved=bool((output_storage[m*inter:]==17.).all()))
                    if not result['candidate_vs_fp32']['passed']:
                        # Retain bounded independent failure evidence. A fast
                        # incorrect output is never a valid candidate.
                        delta=(output.float()-reference.float()).abs()
                        wrong=delta>case_atol+.02*reference.float().abs()
                        points=wrong.nonzero()[:8].cpu().tolist()
                        evidence=[]
                        for row,column in points:
                            pair=(weight[[column,inter+column]].cpu().float()
                                  @x[row].cpu().float()).to(torch.bfloat16).float()
                            cpu_value=(F.silu(pair[0])*pair[1]).to(x.dtype).item()
                            evidence.append(dict(row=row,column=column,
                                candidate=output[row,column].item(),native=baseline[row,column].item(),
                                fp32=reference[row,column].item(),cpu=cpu_value,
                                rounded_gate=pair[0].item(),rounded_up=pair[1].item()))
                        result['failure_point_evidence']=evidence
                    timings=[[],[]]
                    for repeat in range(7):
                        for idx in ((0,1) if repeat%2==0 else (1,0)):
                            selected=(graph_control,graph)[idx]
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(50):selected.replay()
                            end.record();end.synchronize()
                            timings[idx].append(start.elapsed_time(end)/50)
                    before,after=map(statistics.median,timings)
                    result.update(baseline_ms=before,candidate_ms=after,
                                  time_change_percent=100*(after/before-1),raw_ms=timings)
                except Exception as error:
                    result['compile_or_runtime_error']=str(error)
                data['cases'].append(result);save()
                print(json.dumps({key:value for key,value in result.items() if key not in ['raw_ms','cpu_reference_checks']}),flush=True)
                del graph,output,output_storage
            del graph_control,baseline,storage,projected,x,weight
            torch.cuda.empty_cache()
    data.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    save()


if __name__=='__main__':
    main()
