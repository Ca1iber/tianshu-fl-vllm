# SPDX-License-Identifier: Apache-2.0
"""Separate M-dependent W layout, grouping, quantization, GEMM and SiLU fusion."""
import argparse,hashlib,importlib.util,json,os,signal,statistics,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
ROWS=[1,8,16,32,48,64,96,128,192,256,384,512,576,768,1024,1152,1536,1664,1920,2048,2560,3072,3584,4096]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output.resolve()
    seconds=int((datetime.fromisoformat('2026-10-21T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert seconds>1200;signal.alarm(min(seconds,5400))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['CPU_only']
    assert all(sha(f)==d for f,d in gate['sources'].items())
    assert root.is_relative_to('/workspace/benchmark-results');root.mkdir(exist_ok=False);(root/'hardware_before.txt').write_text(hardware)
    state=dict(complete=False,phase='starting',runner_pid=os.getpid(),component_only=True,model_executed=False,adopted=False,rows=ROWS,cases=[],stage_timings=[],sources=gate['sources'])
    def save():
        t=root/'result.tmp';t.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');t.replace(root/'result.json')
    save()
    try:
        import torch
        from safetensors import safe_open
        from flag_gems.fused.silu_and_mul import silu_and_mul_out
        torch.set_num_threads(8)
        nn=load('_range_nn',EXP/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py')
        nt=load('_range_nt',EXP/'tianshu_dense_w8a8_20261020/kernel.py')
        fused=load('_range_fused',EXP/'tianshu_m_range_stages_20261021/int8_fused_grouped.py')
        for f,d in gate['sources'].items():
            dest=root/'source_snapshot'/Path(f).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(f).read_bytes());assert sha(dest)==d
        def fingerprint(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
        def pack(x):
            v=x.float();s=v.abs().amax(-1).clamp_min(1.e-8)/127.;return (v/s[:,None]).round().clamp(-127,127).to(torch.int8),s
        def meta(k):return dict(hash=k.hash,binary_sha256=hashlib.sha256(k.kernel).hexdigest(),registers=k.n_regs,spills=k.n_spills)
        def guarded(shape,dtype,value):
            size=1
            for v in shape:size*=v
            x=torch.full((size+512,),value,device='cuda',dtype=dtype);v=x[256:256+size].view(shape);assert v.data_ptr()%256==0
            return x,v,value
        def guards(buffers):
            for x,_,v in buffers:assert bool((x[:256]==v).all() and (x[-256:]==v).all())
        def timing(g):
            x,y=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);x.record()
            for _ in range(20):g.replay()
            y.record();y.synchronize();return x.elapsed_time(y)/20
        def graph(fn):
            for _ in range(2):fn()
            torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):fn()
            return g
        plan=json.loads((EXP/'tianshu_prefill_int8_service_20261021/selected_plan.json').read_text())['configurations']
        warm=json.loads((EXP/'tianshu_m_range_stages_20261021/OBSERVED_M_32g.json').read_text())['warm_M']
        with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as f:
            wc=torch.cat([f.get_tensor('model.layers.0.mlp.'+part+'_proj.weight') for part in ('gate','up')],0).contiguous()
        wqc,wsc=pack(wc);n,k=wc.shape;i=n//2;assert (n,k)==(12288,2048)
        signature=next(r for r in json.loads((EXP/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text()) if r['prefix']=='model.layers.0.mlp.gate_up_proj')
        assert all(fingerprint(t)==signature[key] for t,key in [(wc,'original_BF16_sha256'),(wqc,'packed_sha256'),(wsc,'scales_sha256')])
        state['weight_signature']=signature;state['static_extra_transposed_W_bytes']=wqc.numel();state['extra_W_if_reuse_existing_NN_bank_bytes']=0
        wn=wqc.cuda();wt=wqc.T.contiguous().cuda();ws=wsc.cuda()
        configs=[dict(bm=bm,bn=bn,bk=128,warps=4 if bn==64 else 8,stages=1,mode='two_dot',group_m=g) for bm in (64,128) for bn in (64,128) for g in (1,8)]
        with torch.inference_mode():
          for m in ROWS:
            ac=torch.randn(m,k,dtype=torch.bfloat16,generator=torch.Generator().manual_seed(202610211800+m))
            if m>1:ac[0].zero_()
            if m>1:ac[1].zero_();ac[1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16)
            a=ac.cuda();wn.copy_(wqc);wt.copy_(wqc.T.contiguous());ws.copy_(wsc);buffers=[];d={}
            for name,shape,dtype,value in [('q',(m,k),torch.int8,17),('s',(m,),torch.float32,17.),('base_proj',(m,n),torch.bfloat16,17.),('base_int',(m,n),torch.int32,-777),('base_out',(m,i),torch.bfloat16,17.),('nt_proj',(m,n),torch.bfloat16,17.),('nt_int',(m,n),torch.int32,-777),('nt_out',(m,i),torch.bfloat16,17.),('cand_proj',(m,n),torch.bfloat16,17.),('cand_int',(m,n),torch.int32,-777),('cand_out',(m,i),torch.bfloat16,17.),('current_out',(m,i),torch.bfloat16,17.)]:
                buf=guarded(shape,dtype,value);buffers.append(buf);d[name]=buf[1]
            if m in (8,64):base_cfg=dict(bm=32 if m==8 else 64,bn=256,bk=128,warps=4,stages=1,group_m=1)
            else:base_cfg=next((r['configuration'] for r in plan if r['M']>=m),plan[-1]['configuration'])
            current_cfg=dict(bm=64,bn=64,bk=128,warps=4,stages=1,mode='two_dot',group_m=1)
            def quant():return nn.quantize(a,d['q'],d['s'])
            def projection():return nn.mm(d['q'],wt,d['s'],ws,d['base_proj'],d['base_int'],base_cfg)
            def activation():return silu_and_mul_out(d['base_proj'][:,:i],d['base_proj'][:,i:],d['base_out'])
            def baseline():quant();projection();activation()
            def nt_total():
                quant();nt.mm(d['q'],wn,d['s'],ws,d['nt_proj'],d['nt_int'],base_cfg);silu_and_mul_out(d['nt_proj'][:,:i],d['nt_proj'][:,i:],d['nt_out'])
            def current():
                if m==64:quant();fused.fused(d['q'],wt,d['s'],ws,d['current_out'],d['cand_int'],d['cand_proj'],current_cfg)
                else:baseline()
            baseline();current_graph=graph(current);base_graph=graph(baseline);nt_graph=graph(nt_total)
            stages={'quantize_only':graph(quant),'NN_GEMM_only':graph(projection),'activation_only':graph(activation),'NN_total':base_graph,'NT_same_tile_total':nt_graph,'current_total':current_graph}
            stage_times={name:[] for name in stages}
            for trial in range(7):
                order=list(stages);order=order[trial%len(order):]+order[:trial%len(order)]
                for name in order:stage_times[name].append(timing(stages[name]))
            state['stage_timings'].append(dict(M=m,baseline_configuration=base_cfg,current_model_INT8_enabled=m in (8,64) or m in warm,M64_current_already_fused=m==64,times_ms=stage_times))
            rows=torch.linspace(0,m-1,min(m,8)).long();cols=torch.linspace(0,i-1,16).long();full_cols=torch.cat((cols,cols+i));rd,cd,fc=rows.cuda(),cols.cuda(),full_cols.cuda()
            for phase in range(3):
                actual=ac if phase==0 else ac*(.5 if phase==1 else -.5);sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.
                aq,asc=pack(actual)
                torch.save(dict(A=actual,q=aq,scale=asc,rows=rows,columns=full_cols,W=wqc[full_cols]*sign,W_scale=wsc[full_cols]*factor),root/f'fixture_M{m}_phase{phase}.pt')
            for ci,cfg in enumerate(configs):
                case=dict(M=m,config_id=ci,configuration=cfg,baseline_configuration=base_cfg,phases=[],complete=False,current_model_INT8_enabled=m in (8,64) or m in warm);state['cases'].append(case);save()
                a.copy_(ac);wn.copy_(wqc);wt.copy_(wqc.T.contiguous());ws.copy_(wsc)
                def invoke():quant();return fused.fused(d['q'],wt,d['s'],ws,d['cand_out'],d['cand_int'],d['cand_proj'],cfg)
                def fused_only():return fused.fused(d['q'],wt,d['s'],ws,d['cand_out'],d['cand_int'],d['cand_proj'],cfg)
                compiled=invoke();case['actual_timed_kernel']=meta(compiled);candidate_graph=graph(invoke);fused_graph=graph(fused_only)
                for phase in range(3):
                    actual=ac if phase==0 else ac*(.5 if phase==1 else -.5);sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.
                    a.copy_(actual);wn.copy_(wqc*sign);wt.copy_((wqc*sign).T.contiguous());ws.copy_(wsc*factor)
                    base_graph.replay();nt_graph.replay();candidate_graph.replay();current_graph.replay();torch.cuda.synchronize()
                    assert torch.equal(d['base_out'].view(torch.int16),d['cand_out'].view(torch.int16)) and torch.equal(d['base_out'].view(torch.int16),d['nt_out'].view(torch.int16)),(m,ci,phase,'timed outputs')
                    if m==64:assert torch.equal(d['base_out'].view(torch.int16),d['current_out'].view(torch.int16))
                    assert bool(torch.isfinite(d['base_out']).all())
                    nn.mm(d['q'],wt,d['s'],ws,d['base_proj'],d['base_int'],base_cfg,return_int=True)
                    nt.mm(d['q'],wn,d['s'],ws,d['nt_proj'],d['nt_int'],base_cfg,return_int=True)
                    fused.fused(d['q'],wt,d['s'],ws,d['cand_out'],d['cand_int'],d['cand_proj'],cfg,diagnostics=True)
                    assert torch.equal(d['base_int'],d['cand_int']) and torch.equal(d['base_int'],d['nt_int'])
                    assert torch.equal(d['base_proj'].view(torch.int16),d['cand_proj'].view(torch.int16)) and torch.equal(d['base_proj'].view(torch.int16),d['nt_proj'].view(torch.int16))
                    aq,asc=pack(actual);assert torch.equal(d['q'].cpu(),aq) and torch.equal(d['s'].cpu(),asc)
                    dot=aq[rows].to(torch.int64)@(wqc[full_cols]*sign).to(torch.int64).T
                    selected=d['base_int'][rd[:,None],fc[None,:]].cpu();assert torch.equal(selected.to(torch.int64),dot)
                    expected=(dot.float()*asc[rows,None]*(wsc[full_cols]*factor)[None,:]).to(torch.bfloat16)
                    assert torch.equal(d['base_proj'][rd[:,None],fc[None,:]].cpu().view(torch.int16),expected.view(torch.int16))
                    gate,up=expected[:,:16].float(),expected[:,16:].float();act=gate/(1+torch.exp(-gate))*up;actual_out=d['cand_out'][rd[:,None],cd[None,:]].cpu()
                    assert ((actual_out.float()-act).abs()<=.001+.02*act.abs()).all()
                    assert torch.equal(a.cpu(),actual) and torch.equal(wt.cpu(),(wqc*sign).T.contiguous()) and torch.equal(ws.cpu(),wsc*factor);guards(buffers)
                    artifact=f'M{m}_cfg{ci}_phase{phase}.pt'
                    torch.save(dict(integer=selected,projected=d['cand_proj'][rd[:,None],fc[None,:]].cpu(),output=actual_out),root/artifact)
                    case['phases'].append(dict(phase=phase,artifact=artifact,full_timed_BF16_bitwise=True,full_INT32_projection_bitwise=True,full_quantization_exact=True,CPU_selected_INT64_exact=True,CPU_activation_tolerance=True,guards_inputs_unchanged=True))
                graphs={'current_total':current_graph,'NN_unfused_total':base_graph,'candidate_total':candidate_graph,'candidate_no_quant':fused_graph};times={name:[] for name in graphs}
                for trial in range(7):
                    order=list(graphs);order=order[trial%4:]+order[:trial%4]
                    if trial%2:order.reverse()
                    for name in order:times[name].append(timing(graphs[name]))
                guards(buffers);base_graph.replay();candidate_graph.replay();torch.cuda.synchronize();assert torch.equal(d['base_out'].view(torch.int16),d['cand_out'].view(torch.int16))
                case.update(complete=True,times_ms=times,medians_ms={name:statistics.median(t) for name,t in times.items()},timed_outputs_after_timing_bitwise=True)
                save();print(json.dumps(dict(M=m,config_id=ci,configuration=cfg,medians_ms=case['medians_ms'])),flush=True)
                del candidate_graph,fused_graph
            del stages,graphs,base_graph,nt_graph,current_graph,d,buffers,a;torch.cuda.empty_cache()
        assert all(sha(f)==digest for f,digest in gate['sources'].items());state.update(complete=True,phase='complete');save()
    except BaseException as error:state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
