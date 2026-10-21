# SPDX-License-Identifier: Apache-2.0
"""Diagnose paired-column loading with unchanged math and guarded controls."""
import hashlib,importlib.util,json,signal,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261021_nn_silu_load_debug_GPU_tianshu')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    assert not ROOT.exists();ROOT.mkdir();signal.alarm(600)
    sources=read(BASE/'CPU_GATE_32g.json')['sources']
    for p,h in sources.items():
        assert sha(p)==h;dest=ROOT/'source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes())
    hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(ROOT/'hardware_before.txt').write_text(hw)
    import torch
    from safetensors import safe_open
    from flag_gems.fused.silu_and_mul import silu_and_mul_out
    native=load('_fused_int8_native_GEMM',BASE.parent/'tianshu_int8_weight_nn_20261021/kernel_nn.py');fused=load('_fused_int8_GateUp_SiLU',BASE/'kernel.py');torch.set_num_threads(8)
    with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as ck:wc=torch.cat((ck.get_tensor('model.layers.0.mlp.gate_proj.weight'),ck.get_tensor('model.layers.0.mlp.up_proj.weight')),0).contiguous()
    f=wc.float();ws_cpu=f.abs().amax(1).clamp_min(1.e-8)/127.;wq=(f/ws_cpu[:,None]).round().clamp(-127,127).to(torch.int8)
    def tensor_sha(x):return hashlib.sha256(memoryview(x.contiguous().view(torch.uint8).numpy())).hexdigest()
    sig=next(r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json') if r['prefix']=='model.layers.0.mlp.gate_up_proj')
    assert tensor_sha(wc)==sig['original_BF16_sha256'] and tensor_sha(wq)==sig['packed_sha256'] and tensor_sha(ws_cpu)==sig['scales_sha256']
    gen=torch.Generator().manual_seed(202610210322+800);ac=torch.randn(8,2048,dtype=torch.bfloat16,generator=gen);ac[0].zero_();ac[1].zero_();ac[1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16)
    a=ac.cuda();w=wq.T.contiguous().cuda();ws=ws_cpu.cuda();buffers=[]
    def guard(shape,dtype,value):
        count=1
        for n in shape:count*=n
        store=torch.full((count+512,),value,device='cuda',dtype=dtype);view=store[256:256+count].view(shape);assert view.data_ptr()%256==0;buffers.append((store,value));return view
    q=guard((8,2048),torch.int8,17);asc=guard((8,),torch.float32,17.);rp=guard((8,12288),torch.bfloat16,17.);ri=guard((8,12288),torch.int32,-777);ro=guard((8,6144),torch.bfloat16,17.)
    dp=guard((8,12288),torch.bfloat16,17.);di=guard((8,12288),torch.int32,-777);actual=guard((8,6144),torch.bfloat16,17.);dout=guard((8,6144),torch.bfloat16,17.)
    control=dict(bm=32,bn=256,bk=128,warps=4,stages=1)
    def original():
        native.quantize(a,q,asc);native.mm(q,w,asc,ws,rp,ri,control);silu_and_mul_out(rp[:,:6144],rp[:,6144:],ro)
    for _ in range(3):original()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):original()
    graph.replay();torch.cuda.synchronize();native_out=ro.cpu().clone();native.mm(q,w,asc,ws,rp,ri,control,return_int=True)
    data=dict(A=ac,Q=q.cpu(),A_scale=asc.cpu(),W_NN=w.cpu(),W_scale=ws.cpu(),native_integer=ri.cpu(),native_projected=rp.cpu(),native_timed=native_out,variants={});metadata={}
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();before_w=w.clone()
    for mode in ('two_dot','single_dot','single_cap','single_slabs'):
        cfg=dict(bm=32,bn=64,bk=128,warps=2,stages=1,mode=mode)
        def candidate():
            native.quantize(a,q,asc);return fused.fused(q,w,asc,ws,actual,di,dp,cfg)
        for _ in range(3):compiled=candidate()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):candidate()
        g.replay();torch.cuda.synchronize();saved_actual=actual.cpu().clone()
        debug=fused.fused(q,w,asc,ws,dout,di,dp,cfg,diagnostics=True);torch.cuda.synchronize()
        data['variants'][mode]=dict(actual_timed=saved_actual,diagnostic_integer=di.cpu().clone(),diagnostic_projected=dp.cpu().clone(),diagnostic_output=dout.cpu().clone())
        metadata[mode]=dict(actual_binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),diagnostic_binary_sha256=hashlib.sha256(debug.kernel).hexdigest(),actual_regs=compiled.n_regs,actual_spills=compiled.n_spills,
             integer_differences=int((di!=ri).sum()),projection_differences=int((dp.view(torch.int16)!=rp.view(torch.int16)).sum()),actual_output_differences=int((saved_actual.view(torch.int16)!=native_out.view(torch.int16)).sum()))
        print(json.dumps(dict(mode=mode,**metadata[mode])),flush=True)
        del g,candidate
    torch.save(data,ROOT/'stages.pt')
    assert torch.equal(w,before_w) and torch.equal(a.cpu(),ac) and torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
    assert all(bool((store[:256]==value).all() and (store[-256:]==value).all()) for store,value in buffers)
    state=dict(complete=True,diagnostic_only=True,adopted=False,sources=sources,raw_root=str(ROOT),payload_sha256=sha(ROOT/'stages.pt'),weight_signature=sig,variants=metadata,guards_inputs_RNG_preserved=True)
    (ROOT/'result.json').write_text(json.dumps(state,indent=2)+'\n');signal.alarm(0)
if __name__=='__main__':main()
