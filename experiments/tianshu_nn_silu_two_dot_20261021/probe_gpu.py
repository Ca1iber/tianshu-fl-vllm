# SPDX-License-Identifier: Apache-2.0
"""Real layer-0 NN W8, actual M quantization, declared FlagGems SiLU component."""
import argparse,gc,hashlib,importlib.util,json,signal,statistics,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent
from choices import CASES,configurations,baseline

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args()
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200;signal.alarm(min(remaining,1800))
    state=dict(complete=False,phase='starting',component_only=True,adopted=False,model_quality_not_evaluated=True,cases=[],timing_rounds=[])
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'result.json')
    save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
        gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'];sources=dict(gate['sources']);sources[str(BASE/'CPU_GATE_32g.json')]=sha(BASE/'CPU_GATE_32g.json')
        assert all(sha(path)==digest for path,digest in sources.items())
        for path,digest in sources.items():
            dest=root/'source_snapshot'/Path(path).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(path).read_bytes());assert sha(dest)==digest
        import torch
        from safetensors import safe_open
        from flag_gems.fused.silu_and_mul import silu_and_mul_out
        native=load('_fused_int8_native_GEMM',BASE.parent/'tianshu_int8_weight_nn_20261021/kernel_nn.py')
        fused=load('_fused_int8_GateUp_SiLU',BASE/'kernel.py')
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        state.update(phase='probing',frozen_sources=sources,torch_version=torch.__version__,triton_version=fused.triton.__version__,
            native_scalar_cache_key=fused._native_silu_scalar.cache_key,scope='current NN A quantize + NN GateUp + actual FlagGems SiLU versus quantize + fused NN; original compiled-model activation is not assumed equivalent')
        def tensor_sha(x):return hashlib.sha256(memoryview(x.contiguous().view(torch.uint8).numpy())).hexdigest()
        def pack(x):
            xf=x.float();scale=xf.abs().amax(-1).clamp_min(1.e-8)/127.
            return (xf/scale[...,None]).round().clamp(-127,127).to(torch.int8),scale
        def guarded(shape,dtype,value):
            count=1
            for size in shape:count*=size
            store=torch.full((count+512,),value,device='cuda',dtype=dtype);view=store[256:256+count].view(shape)
            assert view.data_ptr()%256==0;return store,view
        def metadata(k):return dict(hash=k.hash,binary_sha256=hashlib.sha256(k.kernel).hexdigest(),n_regs=getattr(k,'n_regs',None),n_spills=getattr(k,'n_spills',None),shared=getattr(k.metadata,'shared',None))
        def measure(graph):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
            for _ in range(20):graph.replay()
            end.record();end.synchronize();return start.elapsed_time(end)/20
        with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as checkpoint:
            wc=torch.cat([checkpoint.get_tensor('model.layers.0.mlp.gate_proj.weight'),checkpoint.get_tensor('model.layers.0.mlp.up_proj.weight')],0).contiguous()
        wqc,wsc=pack(wc);n,k=wc.shape;i=n//2;assert (n,k,i)==(12288,2048,6144)
        signature=next(r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json') if r['prefix']=='model.layers.0.mlp.gate_up_proj')
        proof=dict(original_BF16_sha256=tensor_sha(wc),packed_sha256=tensor_sha(wqc),scales_sha256=tensor_sha(wsc))
        assert all(proof[key]==signature[key] for key in proof);state['weight_signature']=proof
        wqcn=wqc.T.contiguous();wq,ws=wqcn.cuda(),wsc.cuda();generator=torch.Generator().manual_seed(202610210322)
        with torch.inference_mode():
          for m,strided in CASES:
            target=m
            for cfg in configurations(m):
                bm,bn,warps=cfg['bm'],cfg['bn'],cfg['warps']
                baseline_cfg=baseline(m)
                case=dict(M=m,padded_M=target,strided=strided,configuration=cfg,baseline_configuration=baseline_cfg,phases=[],phase_files=[]);state['cases'].append(case);save()
                generator.manual_seed(202610210322+100*m+int(strided));ac=torch.randn(m,k,dtype=torch.bfloat16,generator=generator);ac[0].zero_();ac[1].zero_();ac[1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16)
                astore=torch.full((target,k+(7 if strided else 0)),17.,device='cuda',dtype=torch.bfloat16);a=astore[:m,:k];a.copy_(ac)
                arrays={};buffers=[]
                for field,shape,dtype,value in [('q',(target,k),torch.int8,17),('s',(target,),torch.float32,17.),
                    ('native_proj',(target,n),torch.bfloat16,17.),('native_int',(target,n),torch.int32,-777),('native_out',(target,i),torch.bfloat16,17.),
                    ('fused_proj',(target,n),torch.bfloat16,17.),('fused_int',(target,n),torch.int32,-777),('fused_out',(target,i),torch.bfloat16,17.)]:
                    store,view=guarded(shape,dtype,value);arrays[field]=view;buffers.append((store,value))
                q,s=arrays['q'],arrays['s']
                def invoke_native():
                    native.quantize(a,q,s);compiled=native.mm(q,wq,s,ws,arrays['native_proj'],arrays['native_int'],baseline_cfg)
                    silu_and_mul_out(arrays['native_proj'][:,:i],arrays['native_proj'][:,i:],arrays['native_out']);return compiled
                def invoke_fused():
                    native.quantize(a,q,s);return fused.fused(q,wq,s,ws,arrays['fused_out'],arrays['fused_int'],arrays['fused_proj'],cfg)
                wq.copy_(wqcn);ws.copy_(wsc)
                try:
                    for _ in range(3):native_kernel=invoke_native();fused_kernel=invoke_fused()
                except Exception as error:
                    if type(error).__name__ not in ('OutOfResources','CompilationError'):raise
                    case.update(compile_failed=True,exception_type=type(error).__name__,compile_error=str(error),complete=False)
                    torch.cuda.synchronize();save()
                    del arrays,buffers,q,s,store,view,astore,a,ac,invoke_native,invoke_fused
                    gc.collect();torch.cuda.empty_cache()
                    continue
                native_graph,fused_graph=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
                with torch.cuda.graph(native_graph):invoke_native()
                with torch.cuda.graph(fused_graph):invoke_fused()
                case['compiled_native']=metadata(native_kernel);case['compiled_fused_actual_timed']=metadata(fused_kernel)
                columns=torch.linspace(0,i-1,16).long();full_columns=torch.cat((columns,columns+i));rows=torch.linspace(0,m-1,8).long();cd,rd=columns.cuda(),rows.cuda();fc=full_columns.cuda()
                for phase in range(3):
                    actual_a=ac if phase==0 else ac*(.5 if phase==1 else -.5);sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.
                    a.copy_(actual_a);wq.copy_(wqcn*sign);ws.copy_(wsc*factor)
                    before=astore.clone();before_w=wq.clone();before_ws=ws.clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
                    native_graph.replay();fused_graph.replay();torch.cuda.synchronize()
                    assert torch.equal(arrays['native_out'].view(torch.int16),arrays['fused_out'].view(torch.int16)),(m,cfg,phase,'actual timed output')
                    assert bool(torch.isfinite(arrays['native_out']).all())
                    timed_native=arrays['native_out'].cpu().clone();timed_fused=arrays['fused_out'].cpu().clone()
                    native.mm(q,wq,s,ws,arrays['native_proj'],arrays['native_int'],baseline_cfg,return_int=True)
                    debug=fused.fused(q,wq,s,ws,arrays['fused_out'],arrays['fused_int'],arrays['fused_proj'],cfg,diagnostics=True)
                    case['compiled_fused_diagnostic']=metadata(debug)
                    assert torch.equal(arrays['native_int'],arrays['fused_int'])
                    assert torch.equal(arrays['native_proj'].view(torch.int16),arrays['fused_proj'].view(torch.int16))
                    assert torch.equal(arrays['native_out'].view(torch.int16),arrays['fused_out'].view(torch.int16))
                    aq,asc=pack(actual_a)
                    assert torch.equal(q[:m].cpu(),aq) and torch.equal(s[:m].cpu(),asc)
                    dot=aq[rows].to(torch.int64)@(wqc[full_columns]*sign).to(torch.int64).T
                    actual_int=arrays['native_int'][rd[:,None],fc[None,:]].cpu()
                    assert torch.equal(dot,actual_int.to(torch.int64))
                    expected_proj=(dot.float()*asc[rows,None]*(wsc[full_columns]*factor)[None,:]).to(torch.bfloat16)
                    native_proj=arrays['native_proj'][rd[:,None],fc[None,:]].cpu();fused_proj=arrays['fused_proj'][rd[:,None],fc[None,:]].cpu()
                    assert torch.equal(native_proj.view(torch.int16),expected_proj.view(torch.int16))
                    cpu_gate,cpu_up=expected_proj[:,:16].float(),expected_proj[:,16:].float()
                    cpu_activated=cpu_gate/(1.+torch.exp(-cpu_gate))*cpu_up
                    actual=arrays['native_out'][rd[:,None],cd[None,:]].cpu()
                    assert bool(torch.isfinite(actual).all() and ((actual.float()-cpu_activated).abs()<=.001+.02*cpu_activated.abs()).all())
                    assert torch.equal(astore,before) and torch.equal(wq,before_w) and torch.equal(ws,before_ws)
                    assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
                    for store,value in buffers:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
                    filename=f"M{m}_strided{int(strided)}_{cfg['mode']}_bm{bm}_bn{bn}_w{warps}_phase{phase}.pt"
                    torch.save(dict(phase=phase,weight_sign=sign,weight_scale_factor=factor,A=actual_a,A_INT8=aq,A_scale=asc,
                        original_W_rows=wc[full_columns].clone(),W_INT8=wqc[full_columns]*sign,W_scale=wsc[full_columns]*factor,rows=rows,columns=full_columns,
                        native_integer=actual_int,fused_integer=arrays['fused_int'][rd[:,None],fc[None,:]].cpu(),
                        native_projected=native_proj,fused_projected=fused_proj,native_output=actual,
                        full_timed_native_output=timed_native,full_timed_fused_output=timed_fused,fused_output=arrays['fused_out'][rd[:,None],cd[None,:]].cpu()),root/filename)
                    case['phase_files'].append(filename);case['phases'].append(dict(phase=phase,full_actual_timed_BF16_output_bitwise=True,
                        full_diagnostic_INT32_and_projected_BF16_bitwise=True,full_quantization_exact=True,CPU_selected_INT64_and_projection_exact=True,
                        CPU_selected_activation_tolerance_passed=True,all_guards_inputs_RNG_preserved=True,no_actual_M_padding=True))
                    del before,before_w,before_ws
                for trial in range(7):
                    order=['native','fused'] if trial%2==0 else ['fused','native'];graphs={'native':native_graph,'fused':fused_graph}
                    times={name:measure(graphs[name]) for name in order}
                    state['timing_rounds'].append(dict(M=m,padded_M=target,strided=strided,configuration=cfg,round=trial,replays=20,order=order,native_ms=times['native'],fused_ms=times['fused']))
                for store,value in buffers:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
                case['complete']=True;case['timing_guards_passed']=True;save();print(json.dumps(dict(M=m,configuration=cfg,complete=True)),flush=True)
                del arrays,buffers,q,s,store,view,astore,a,ac,native_graph,fused_graph,graphs,invoke_native,invoke_fused
                gc.collect();torch.cuda.empty_cache()
        assert len(state['cases'])==16 and all(sha(path)==digest for path,digest in sources.items())
        state.update(complete=True,phase='complete');save()
    except BaseException as error:
        import traceback
        state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
