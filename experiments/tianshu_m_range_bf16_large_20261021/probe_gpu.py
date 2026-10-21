# SPDX-License-Identifier: Apache-2.0
"""Role-specific tile/layout controls retaining the current M2048 projection path."""
import argparse,hashlib,importlib.util,json,os,signal,statistics,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
ROWS=[1536,1664,1920,2048,2304,2560,2816,3072,3328,3584,3840,4096]
CONFIGS=[dict(bm=bm,bn=bn,bk=bk,warps=warps,stages=2,group_m=g) for bm,bn,bk,warps in [(128,128,32,8),(128,256,32,8),(256,128,32,8),(256,256,32,16),(128,256,64,8),(128,128,64,8),(256,256,64,16)] for g in (1,8)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output.resolve()
    remaining=int((datetime.fromisoformat('2026-10-21T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert remaining>900;signal.alarm(min(remaining,5400))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['CPU_only'];assert all(sha(f)==d for f,d in gate['sources'].items())
    assert root.is_relative_to('/workspace/benchmark-results');root.mkdir(exist_ok=False);(root/'hardware_before.txt').write_text(hardware)
    state=dict(complete=False,phase='starting',runner_pid=os.getpid(),component_only=True,model_executed=False,adopted=False,rows=ROWS,configurations=CONFIGS,sources=gate['sources'],cases=[])
    def save():
        p=root/'result.tmp';p.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');p.replace(root/'result.json')
    save()
    try:
        import torch,flag_gems,triton
        from safetensors import safe_open
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        matched=load('_bf16_range_matched',EXP/'tianshu_m_range_stages_20261021/matched.py')
        current=load('_bf16_current_2048',EXP/'tianshu_prefill_gemm_tiles_20261014/experimental_mm.py')
        for f,d in gate['sources'].items():
            dest=root/'source_snapshot'/Path(f).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(f).read_bytes());assert sha(dest)==d
        def check(x,y):
            d=(x.float()-y.float()).abs();bad=d>.02+.01*y.float().abs();return dict(passed=bool(torch.isfinite(x).all() and torch.isfinite(y).all() and not bad.any()),bad=int(bad.sum()),max_abs=float(d.max()))
        def timing(g):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(20):g.replay()
            b.record();b.synchronize();return a.elapsed_time(b)/20
        with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as checkpoint,torch.inference_mode():
          for role in ('qkv','o','down'):
            keys=(['model.layers.0.self_attn.'+s+'_proj.weight' for s in ('q','k','v')] if role=='qkv' else ['model.layers.0.self_attn.o_proj.weight'] if role=='o' else ['model.layers.0.mlp.down_proj.weight'])
            wc=torch.cat([checkpoint.get_tensor(key) for key in keys],0).contiguous();n,k=wc.shape;w=wc.cuda()
            for ci,cfg in enumerate(CONFIGS):
              bm,bn,bk=cfg['bm'],cfg['bn'],cfg['bk'];start=time.perf_counter();wpc=wc.reshape(n//bn,bn,k//bk,bk).permute(0,2,1,3).contiguous();pack_seconds=time.perf_counter()-start;wp=wpc.cuda()
              for m in ROWS:
                ac=torch.randn(m,k,dtype=torch.bfloat16,generator=torch.Generator().manual_seed(202610212000+m+len(role)));ac[0].zero_();a=ac.cuda();w.copy_(wc);wp.copy_(wpc)
                outputs={name:torch.empty(m,n,dtype=torch.bfloat16,device='cuda') for name in ('native','current','matched_NT','packed_W')};kernels={}
                def raw(label,packed):
                    kernels[label]=matched.matched_nt_kernel[(triton.cdiv(m,bm)*(n//bn),)](a,wp if packed else w,outputs[label],m,n,k,BM=bm,BN=bn,BK=bk,GROUP_M=cfg['group_m'],PACKED=packed,EVEN_M=m%bm==0,EVEN_N=True,num_warps=cfg['warps'],num_stages=2)
                def current_model():
                    if m==2048 and role!='qkv':current.mm(a,w,(outputs['current'],outputs['current']),dict(bm=256,bn=256,bk=32,splits=1,warps=16,stages=2,group_m=1))
                    else:flag_gems.mm_out(a,w.T,out=outputs['current'])
                functions={'native':lambda:flag_gems.mm_out(a,w.T,out=outputs['native']),'current':current_model,'matched_NT':lambda:raw('matched_NT',False),'packed_W':lambda:raw('packed_W',True)};graphs={}
                for label,fn in functions.items():
                    for _ in range(2):fn()
                    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):fn()
                    graphs[label]=g
                for phase in range(2):
                    factor=1. if phase==0 else -.5;a.copy_(ac*factor);w.copy_(wc*factor);wp.copy_(wpc*factor)
                    for g in graphs.values():g.replay()
                    torch.cuda.synchronize();ref=(a.float()@w.float().T).to(torch.bfloat16);before={name:check(out,ref) for name,out in outputs.items()}
                    assert before['matched_NT']['passed'] and before['packed_W']['passed'],(role,ci,m,phase,before)
                    rows=torch.tensor(sorted({0,1,m//2,m-1}));cols=torch.linspace(0,n-1,16).long();rd,cd=rows.cuda(),cols.cuda();sample_a=a[rd].cpu();sample_w=w[cd].cpu();cpu=(sample_a.double()@sample_w.double().T).to(torch.bfloat16)
                    sample_before={name:out[rd[:,None],cd[None,:]].cpu() for name,out in outputs.items()};sample_checks={name:check(x,cpu) for name,x in sample_before.items()}
                    assert sample_checks['matched_NT']['passed'] and sample_checks['packed_W']['passed']
                    times={name:[] for name in graphs}
                    for trial in range(7):
                        order=list(graphs);order=order[trial%4:]+order[:trial%4]
                        if trial%2:order.reverse()
                        for name in order:times[name].append(timing(graphs[name]))
                    after={name:check(out,ref) for name,out in outputs.items()};assert after['matched_NT']['passed'] and after['packed_W']['passed'],(role,ci,m,phase,after)
                    sample_after={name:out[rd[:,None],cd[None,:]].cpu() for name,out in outputs.items()};eligible={name:before[name]['passed'] and after[name]['passed'] and sample_checks[name]['passed'] for name in ('native','current')}
                    filename=f'{role}_cfg{ci}_M{m}_phase{phase}.pt';torch.save(dict(A=sample_a,W=sample_w,CPU_reference=cpu,before=sample_before,after=sample_after),root/filename)
                    if not all(eligible.values()):torch.save(dict(A=a.cpu(),W=w.cpu(),reference=ref.cpu(),outputs={name:out.cpu() for name,out in outputs.items()}),root/filename.replace('.pt','_failed_reference_full.pt'))
                    assert torch.equal(a.cpu(),ac*factor) and torch.equal(w.cpu(),wc*factor)
                    state['cases'].append(dict(role=role,M=m,phase=phase,config_id=ci,configuration=cfg,reference_eligible=eligible,before=before,after=after,selected_checks=sample_checks,times_ms=times,medians_ms={name:statistics.median(values) for name,values in times.items()},artifact=filename,static_weight_pack_CPU_seconds=pack_seconds,extra_W_bytes=wp.numel()*wp.element_size(),compiled={label:dict(hash=kernel.hash,binary_sha256=hashlib.sha256(kernel.kernel).hexdigest(),registers=kernel.n_regs,spills=kernel.n_spills) for label,kernel in kernels.items()}));state['phase']='probing';save()
                    print(json.dumps(dict(role=role,M=m,config_id=ci,phase=phase,reference_eligible=eligible,medians_ms=state['cases'][-1]['medians_ms'])),flush=True)
                del graphs,outputs,a,ref;torch.cuda.empty_cache()
        assert all(sha(f)==d for f,d in gate['sources'].items());state.update(complete=True,phase='complete');save()
    except BaseException as error:state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
