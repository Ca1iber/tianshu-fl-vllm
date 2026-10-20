# SPDX-License-Identifier: Apache-2.0
"""Independent RMS/A8 component; no model hooks or model score claims."""
import argparse,ast,hashlib,importlib.util,json,signal,statistics,time
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent
QUANT=BASE.parent/'tianshu_dense_w8a8_20261020/kernel.py'
CONFIGS=[dict(rows=1,warps=4),dict(rows=4,warps=4),dict(rows=4,warps=8)]
SIZES=(8,64,4096)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def sources():
    paths=[*BASE.glob('*.py'),QUANT,Path('/usr/local/lib/python3.12/site-packages/vllm/ir/ops/layernorm.py'),Path('/workspace/benchmark-results/20261021_down_native_compile_evidence_tianshu/generated_model.py')]
    return {str(p):sha(p) for p in paths}
def normalize(x,residual,weight):
    combined=x.float()+residual.float()
    inv=torch.rsqrt((combined*combined).mean(-1,keepdim=True)+1.e-6)
    return ((combined*inv).to(weight.dtype)*weight).to(x.dtype)
def inputs(m,phase):
    rows=torch.arange(m,dtype=torch.int32)[:,None];cols=torch.arange(2048,dtype=torch.int32)[None,:]
    x=(((rows*37+cols*13+phase*103)%2048-1023)/128.).to(torch.bfloat16)
    r=(((rows*19+cols*29+phase*79)%2048-1023)/256.).to(torch.bfloat16)
    x[0].zero_();r[0].zero_()
    if m>1:r[1]=-x[1]
    return x,r

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-only',action='store_true');parser.add_argument('--out',type=Path);args=parser.parse_args()
    global torch
    import torch
    torch.set_num_threads(4)
    if args.cpu_only:
        assert not torch.cuda.is_initialized()
        for p in BASE.glob('*.py'):ast.parse(p.read_text())
        for m in range(1,130):
            for c in CONFIGS:
                selected=[i for block in range((m+c['rows']-1)//c['rows']) for i in range(block*c['rows'],(block+1)*c['rows']) if i<m]
                assert selected==list(range(m))
        assert max(SIZES)*2048<2**31 and sha(QUANT)=='b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152'
        # Pin the actual original quantizer, without substituting a guessed hash.
        assert 'scale=tl.div_rn' in QUANT.read_text() and 'libdevice.nearbyint' in QUANT.read_text()
        gate=dict(complete=True,CPU_only=True,no_GPU_started=True,sizes=SIZES,configurations=CONFIGS,row_coverage_and_masks_checked=True,int32_output_bound=max(SIZES)*2048,source_sha256=sources(),new_model_Math_required_if_integrated=True)
        (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate));return
    assert args.out and args.out.is_absolute() and args.out.is_relative_to('/workspace/benchmark-results') and not args.out.exists()
    root=args.out;root.mkdir();cpu=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert cpu['complete'] and cpu['source_sha256']==sources()
    deadline=datetime.fromisoformat('2026-10-21T15:58:00+00:00');seconds=int((deadline-datetime.now(timezone.utc)).total_seconds());assert seconds>600
    signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(TimeoutError('reservation hard stop')));signal.alarm(seconds)
    from safetensors import safe_open
    import triton
    kernel=load('_rms_a8_component_kernel',BASE/'kernel.py');quant=load('_rms_a8_qualified_quantizer',QUANT)
    with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as weights:weight=weights.get_tensor('model.layers.0.post_attention_layernorm.weight')
    assert weight.shape==(2048,) and weight.dtype==torch.bfloat16
    weight_sha=hashlib.sha256(weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();w=weight.cuda()
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
    result=dict(phase='running',component_only=True,adopted=False,model_not_installed=True,model_Math_not_run=True,source_sha256=sources(),CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),weight_sha256=weight_sha,cases=[],fixtures={})
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(root/'result.json')
    def fixture(name,data):
        p=root/name;torch.save(data,p);result['fixtures'][name]=dict(sha256=sha(p),bytes=p.stat().st_size);return name
    save()
    compiled=torch.compile(normalize,fullgraph=True,dynamic=False,options={'trace.enabled':True,'trace.debug_dir':str(root/'compile_trace')})
    for m in SIZES:
        xcpu,rcpu=inputs(m,0);x=xcpu.cuda();r=rcpu.cuda();q0=torch.empty((m,2048),device='cuda',dtype=torch.int8);s0=torch.empty(m,device='cuda')
        norm=torch.full((m+1,2048),19.,device='cuda',dtype=torch.bfloat16);q=torch.full((m+1,2048),69,device='cuda',dtype=torch.int8);scales=torch.full((m+1,),-37.,device='cuda')
        row=dict(M=m,phases=[],configurations=[],rounds=[])
        for phase in range(3):
            xc,rc=inputs(m,phase);x.copy_(xc);r.copy_(rc);original_x=x.clone();original_r=r.clone()
            native=compiled(x,r,w);quant.quantize(native,q0,s0);torch.cuda.synchronize()
            baseline_cpu=dict(norm=native.cpu(),q=q0.cpu(),scales=s0.cpu())
            phase_row=dict(phase=phase,variants=[])
            if phase==2:row['baseline_fixture']=fixture(f'baseline_M{m}.pt',dict(x=xc,residual=rc,weight=weight,**baseline_cpu))
            for index,c in enumerate(CONFIGS):
                built=kernel.rms_a8[(triton.cdiv(m,c['rows']),)](x,r,w,norm,q,scales,m,2048,c['rows'],num_warps=c['warps'],num_stages=1)
                torch.cuda.synchronize();n=norm[:m].cpu();qc=q[:m].cpu();sc=scales[:m].cpu();values=n.float();expected_s=values.abs().amax(1).clamp_min(1.e-8)/127.;expected_q=torch.round(values/expected_s[:,None]).clamp(-127,127).to(torch.int8)
                assert torch.equal(sc,expected_s) and torch.equal(qc,expected_q)
                delta=(n.float()-baseline_cpu['norm'].float()).abs();bound=1.e-5+.015625*baseline_cpu['norm'].float().abs()
                assert not bool((delta>bound).any()) and bool(torch.isfinite(n).all())
                qdelta=(qc.short()-baseline_cpu['q'].short()).abs();assert int(qdelta.max())<=2
                assert bool((norm[m]==19.).all()) and bool((q[m]==69).all()) and float(scales[m])==-37.
                check=dict(index=index,configuration=c,full_quantization_and_scales_exact_from_materialized_BF16=True,compiled_norm_max_absolute_error=float(delta.max()),compiled_norm_unequal_elements=int((n!=baseline_cpu['norm']).sum()),compiled_quant_unequal_elements=int((qc!=baseline_cpu['q']).sum()),maximum_quant_difference=int(qdelta.max()),compiled_reference_tolerance_passed=True,guards_preserved=True)
                if phase==2:check['fixture']=fixture(f'candidate_M{m}_config{index}.pt',dict(norm=n,q=qc,scales=sc))
                phase_row['variants'].append(check)
                if phase==0:row['configurations'].append(dict(index=index,configuration=c,binary_sha256=hashlib.sha256(built.kernel).hexdigest(),cache_hash=built.hash,n_regs=built.n_regs,n_spills=built.n_spills,shared=built.metadata.shared))
            assert torch.equal(x,original_x) and torch.equal(r,original_r)
            row['phases'].append(phase_row)
        def baseline():
            a=compiled(x,r,w);quant.quantize(a,q0,s0)
        graphs={}
        for name in ['baseline']+[f'fused_{i}' for i in range(len(CONFIGS))]:
            if name=='baseline':fn=baseline
            else:
                c=CONFIGS[int(name.split('_')[1])]
                def fn(c=c):return kernel.rms_a8[(triton.cdiv(m,c['rows']),)](x,r,w,norm,q,scales,m,2048,c['rows'],num_warps=c['warps'],num_stages=1)
            for _ in range(3):fn()
            torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):fn()
            graphs[name]=g
        names=list(graphs)
        for repeat in range(7):
            order=names[repeat%len(names):]+names[:repeat%len(names)];values={}
            for name in order:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):graphs[name].replay()
                end.record();end.synchronize();values[name]=start.elapsed_time(end)/20
            row['rounds'].append(dict(round=repeat+1,order=order,milliseconds=values,replays=20))
        row['median_ms']={name:statistics.median(v['milliseconds'][name] for v in row['rounds']) for name in names}
        result['cases'].append(row);save();print(json.dumps(dict(M=m,median_ms=row['median_ms'])),flush=True)
        del graphs,x,r,q0,s0,norm,q,scales,native,original_x,original_r
        torch.cuda.empty_cache()
    assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
    assert result['source_sha256']==sources()
    result.update(phase='GPU_complete',inputs_and_CPU_GPU_RNG_preserved=True,full_three_phase_checks=3*3*len(SIZES),CPU_only_recheck_required=True,compiled_trace_sources={str(p):sha(p) for p in (root/'compile_trace').rglob('output_code.py')})
    save();signal.alarm(0)
if __name__=='__main__':main()
