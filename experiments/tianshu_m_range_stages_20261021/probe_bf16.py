# SPDX-License-Identifier: Apache-2.0
"""Stage QKV/O/Down by M; isolated layout controls plus native-library reference."""
import argparse,hashlib,importlib.util,json,os,signal,statistics,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROWS=[1,8,16,32,48,64,96,128,192,256,384,512,576,768,1024,1152,1536,1664,1920,2048,2560,3072,3584,4096]
ROLES={'qkv':(2560,2048),'o':(2048,2048),'down':(2048,6144)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True)
    p.add_argument('--roles',nargs='+',choices=ROLES,default=list(ROLES));args=p.parse_args()
    for pid in (249716,255255):
        status=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
        assert not status or status.startswith('Z'),f'GPU validation still running: PID {pid}'
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>0
    signal.alarm(min(remaining,3600))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'Iluvatar BI-V150' in hardware and 'No running processes found' in hardware
    gate=json.loads((BASE/'CPU_BF16_PREPARATION_32g.json').read_text());assert gate['complete'] and gate['CPU_only']
    for filename,digest in gate['sources'].items():assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results');root.mkdir(exist_ok=False)
    (root/'hardware_before.txt').write_text(hardware)
    state=dict(complete=False,phase='starting',runner_pid=os.getpid(),component_only=True,service_executed=False,adoption_allowed=False,
        frozen_sources=gate['sources'],roles=args.roles,rows=ROWS,scope='QKV/O/Down BF16 stage; native FlagGems library reference plus exactly matched layout controls, not a full model profile',
        full_output_comparison_scope='GPU all elements checked online; saved tensors contain independent CPU samples, not all full outputs',cases=[])
    def save():
        temporary=root/'result.tmp';temporary.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');temporary.replace(root/'result.json')
    save()
    try:
        import torch,flag_gems,triton
        from safetensors import safe_open
        import matched
        sys.path.insert(0,str(BASE.parent))
        from tianshu_m_range_stages_20261021 import packed_bundle as packed
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        def sha_tensor(x):return hashlib.sha256(memoryview(x.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()
        def check(x,y):
            x,y=x.float(),y.float();delta=(x-y).abs();bad=delta>.02+.01*y.abs()
            return dict(passed=bool(torch.isfinite(x).all() and torch.isfinite(y).all() and not bad.any()),
                failed_elements=int(bad.sum()),maximum_absolute_error=float(delta.max()),atol=.02,rtol=.01)
        def metadata(kernel):return dict(cache_hash=kernel.hash,binary_sha256=hashlib.sha256(kernel.kernel).hexdigest(),
            n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None),shared=getattr(kernel.metadata,'shared',None))
        def timed(graph):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(20):graph.replay()
            b.record();b.synchronize();return a.elapsed_time(b)/20
        for filename in gate['sources']:
            actual=Path(filename);dest=root/'source_snapshot'/actual.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(actual.read_bytes())
        with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as checkpoint,torch.inference_mode():
          for role in args.roles:
            n,k=ROLES[role]
            keys=(['model.layers.0.self_attn.'+part+'_proj.weight' for part in ('q','k','v')] if role=='qkv' else
                ['model.layers.0.self_attn.o_proj.weight'] if role=='o' else ['model.layers.0.mlp.down_proj.weight'])
            wc=torch.cat([checkpoint.get_tensor(key) for key in keys],0).contiguous();assert tuple(wc.shape)==(n,k)
            w=wc.cuda();bn,bk=(64,64) if role=='down' else (128,32)
            packing_started=time.perf_counter()
            packed_cpu=wc.reshape(n//bn,bn,k//bk,bk).permute(0,2,1,3).contiguous();packing_CPU_seconds=time.perf_counter()-packing_started;transfer_started=time.perf_counter();wp=packed_cpu.cuda();torch.cuda.synchronize();packing_H2D_seconds=time.perf_counter()-transfer_started
            weight_proof=dict(keys=keys,source_BF16_sha256=sha_tensor(wc),packed_sha256=sha_tensor(packed_cpu),
                additional_packed_bytes=wp.numel()*wp.element_size(),static_weight_pack_CPU_seconds=packing_CPU_seconds,static_weight_copy_H2D_seconds=packing_H2D_seconds,estimated_42_layer_extra_bytes=42*wp.numel()*wp.element_size(),static_packing_time_is_not_per_step=True)
            for m in ROWS:
                generator=torch.Generator().manual_seed(202610210000+m+(1 if role=='down' else 2 if role=='o' else 3))
                ac=torch.randn(m,k,dtype=torch.bfloat16,generator=generator);ac[0].zero_();a=ac.cuda()
                xp=torch.empty(k//bk,m,bk,device='cuda',dtype=torch.bfloat16)
                bm=64 if role=='down' else 128
                config=dict(bm=bm,bn=bn,bk=bk,warps=4 if role=='down' else 8,stages=2,group_m=8)
                output={label:torch.empty(m,n,device='cuda',dtype=torch.bfloat16) for label in ('native_library','matched_NT','packed_W','packed_AW')}
                kernels={}
                def raw(label,is_packed):
                    kernels[label]=matched.matched_nt_kernel[(triton.cdiv(m,bm)*(n//bn),)](a,wp if is_packed else w,output[label],m,n,k,
                        BM=bm,BN=bn,BK=bk,GROUP_M=8,PACKED=is_packed,EVEN_M=m%bm==0,EVEN_N=True,
                        num_warps=config['warps'],num_stages=2)
                def pack_a():xp.copy_(a.reshape(m,k//bk,bk).permute(1,0,2))
                def packed_aw():
                    pack_a()
                    kernels['packed_AW']=packed._packed_nt_kernel[(triton.cdiv(m,bm)*(n//bn),)](xp,wp,output['packed_AW'],m,n,k,
                        BM=bm,BN=bn,BK=bk,GROUP_M=8,PACK_A=True,EVEN_M=m%bm==0,num_warps=config['warps'],num_stages=2)
                functions={'native_library':lambda:flag_gems.mm_out(a,w.T,out=output['native_library']),
                    'matched_NT':lambda:raw('matched_NT',False),'packed_W':lambda:raw('packed_W',True),'packed_AW':packed_aw,'A_pack_only':pack_a}
                graphs={}
                for label,fn in functions.items():
                    for _ in range(2):fn()
                    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):fn()
                    graphs[label]=graph
                for phase in range(2):
                    factor=1. if phase==0 else -.5
                    a.copy_(ac*(1. if phase==0 else -.5));w.copy_(wc*factor);wp.copy_(packed_cpu*factor)
                    for graph in graphs.values():graph.replay()
                    torch.cuda.synchronize()
                    reference=(a.float()@w.float().T).to(torch.bfloat16)
                    checks={label:check(value,reference) for label,value in output.items()}
                    rows=torch.tensor(sorted({0,1 if m>1 else 0,m//2,m-1}));columns=torch.linspace(0,n-1,16).long()
                    sample_a=a[rows.cuda()].cpu();sample_w=w[columns.cuda()].cpu()
                    expected=(sample_a.double()@sample_w.double().T).to(torch.bfloat16)
                    samples={label:value[rows.cuda()[:,None],columns.cuda()[None,:]].cpu() for label,value in output.items()}
                    cpu={label:check(value,expected) for label,value in samples.items()}
                    payload=dict(role=role,M=m,phase=phase,A=sample_a,W=sample_w,rows=rows,columns=columns,CPU_reference=expected,outputs_before_timing=samples)
                    torch.save(payload,root/f'{role}_M{m}_phase{phase}.pt')
                    assert all(c['passed'] for c in checks.values()),(role,m,phase,checks)
                    assert all(c['passed'] for c in cpu.values()),(role,m,phase,cpu)
                    times={label:[] for label in graphs};labels=list(graphs)
                    for trial in range(7):
                        order=labels[trial%len(labels):]+labels[:trial%len(labels)]
                        if trial%2:order=list(reversed(order))
                        for label in order:times[label].append(timed(graphs[label]))
                    after={label:check(value,reference) for label,value in output.items()}
                    payload['outputs_after_timing']={label:value[rows.cuda()[:,None],columns.cuda()[None,:]].cpu() for label,value in output.items()}
                    torch.save(payload,root/f'{role}_M{m}_phase{phase}.pt')
                    assert all(c['passed'] for c in after.values()),(role,m,phase,after)
                    assert torch.equal(a.cpu(),ac*(1. if phase==0 else -.5)) and torch.equal(w.cpu(),wc*factor)
                    row=dict(role=role,M=m,phase=phase,configuration=config,weight_proof=weight_proof,
                        full_output_GPU_checks=checks,timed_full_output_GPU_checks=after,independent_CPU_selected_checks=cpu,
                        actual_device_kernels={key:metadata(value) for key,value in kernels.items()},times_ms=times,
                        median_ms={label:statistics.median(values) for label,values in times.items()},
                        full_output_sha256={label:sha_tensor(value) for label,value in output.items()},elements_checked_per_implementation=m*n)
                    state['cases'].append(row);state['phase']='probing';save()
                    print(json.dumps(dict(role=role,M=m,phase=phase,median_ms=row['median_ms'])),flush=True)
                del graphs,output,a,xp,reference
                torch.cuda.empty_cache()
        state.update(complete=True,phase='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()
    except BaseException as error:
        state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise

if __name__=='__main__':main()
