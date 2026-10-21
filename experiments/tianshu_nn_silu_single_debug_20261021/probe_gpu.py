# SPDX-License-Identifier: Apache-2.0
"""One deterministic failing single-dot case; save every stage before judging."""
import hashlib,importlib.util,json,os,signal,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261021_nn_silu_single_debug_GPU_tianshu')
PARENT=BASE.parent/'tianshu_int8_nn_gateup_silu_20261021'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    assert not ROOT.exists();ROOT.mkdir();signal.alarm(600)
    gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'];sources=gate['sources']
    for p,h in sources.items():
        assert sha(p)==h;dest=ROOT/'source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes())
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(ROOT/'hardware_before.txt').write_text(hardware)
    import torch
    from safetensors import safe_open
    from flag_gems.fused.silu_and_mul import silu_and_mul_out
    native=load('_fused_int8_native_GEMM',BASE.parent/'tianshu_int8_weight_nn_20261021/kernel_nn.py')
    fused=load('_fused_int8_GateUp_SiLU',PARENT/'kernel.py')
    torch.set_num_threads(8)
    with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as ck:
        wc=torch.cat((ck.get_tensor('model.layers.0.mlp.gate_proj.weight'),ck.get_tensor('model.layers.0.mlp.up_proj.weight')),0).contiguous()
    v=wc.float();wscale=v.abs().amax(1).clamp_min(1.e-8)/127.;wq=(v/wscale[:,None]).round().clamp(-127,127).to(torch.int8)
    def tensor_sha(x):return hashlib.sha256(memoryview(x.contiguous().view(torch.uint8).numpy())).hexdigest()
    signature=next(r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json') if r['prefix']=='model.layers.0.mlp.gate_up_proj')
    assert tensor_sha(wc)==signature['original_BF16_sha256'] and tensor_sha(wq)==signature['packed_sha256'] and tensor_sha(wscale)==signature['scales_sha256']
    gen=torch.Generator().manual_seed(202610210322+800);ac=torch.randn(8,2048,dtype=torch.bfloat16,generator=gen);ac[0].zero_();ac[1].zero_();ac[1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16)
    a=ac.cuda();w=wq.T.contiguous().cuda();ws=wscale.cuda();buffers=[]
    def guard(shape,dtype,value):
        count=1
        for n in shape:count*=n
        store=torch.full((count+512,),value,device='cuda',dtype=dtype);view=store[256:256+count].view(shape);assert view.data_ptr()%256==0;buffers.append((store,value));return view
    q=guard((8,2048),torch.int8,17);asc=guard((8,),torch.float32,17.)
    ref_proj=guard((8,12288),torch.bfloat16,17.);ref_int=guard((8,12288),torch.int32,-777);ref=guard((8,6144),torch.bfloat16,17.)
    diag_proj=guard((8,12288),torch.bfloat16,17.);diag_int=guard((8,12288),torch.int32,-777);actual=guard((8,6144),torch.bfloat16,17.);diagnostic=guard((8,6144),torch.bfloat16,17.)
    cfg=dict(bm=32,bn=64,bk=128,warps=2,stages=1,mode='single_dot');control=dict(bm=32,bn=256,bk=128,warps=4,stages=1)
    def original():
        native.quantize(a,q,asc);k=native.mm(q,w,asc,ws,ref_proj,ref_int,control);silu_and_mul_out(ref_proj[:,:6144],ref_proj[:,6144:],ref);return k
    def candidate():
        native.quantize(a,q,asc);return fused.fused(q,w,asc,ws,actual,diag_int,diag_proj,cfg)
    for _ in range(3):original();actual_kernel=candidate()
    graphs=[torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()]
    with torch.cuda.graph(graphs[0]):original()
    with torch.cuda.graph(graphs[1]):candidate()
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();before_w=w.clone()
    graphs[0].replay();graphs[1].replay();torch.cuda.synchronize()
    ref_timed=ref.cpu().clone();actual_timed=actual.cpu().clone()
    native.mm(q,w,asc,ws,ref_proj,ref_int,control,return_int=True)
    diagnostic_kernel=fused.fused(q,w,asc,ws,diagnostic,diag_int,diag_proj,cfg,diagnostics=True);torch.cuda.synchronize()
    # Persist actual Graph outputs separately from the diagnostic specialization.
    data=dict(A=ac,Q=q.cpu(),A_scale=asc.cpu(),W_NN=w.cpu(),W_scale=ws.cpu(),native_integer=ref_int.cpu(),diagnostic_integer=diag_int.cpu(),native_projected=ref_proj.cpu(),diagnostic_projected=diag_proj.cpu(),native_timed=ref_timed,actual_timed=actual_timed,diagnostic_output=diagnostic.cpu())
    torch.save(data,ROOT/'stages.pt')
    assert torch.equal(w,before_w) and torch.equal(a.cpu(),ac)
    assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
    assert all(bool((store[:256]==value).all() and (store[-256:]==value).all()) for store,value in buffers)
    def metadata(k):return dict(hash=k.hash,binary_sha256=hashlib.sha256(k.kernel).hexdigest(),n_regs=k.n_regs,n_spills=k.n_spills)
    failed=read('/workspace/benchmark-results/20261021_int8_nn_gateup_silu_GPU_tianshu/result.json')['cases'][-1]
    assert failed['configuration']==cfg and failed['M']==8 and not failed['strided']
    state=dict(complete=True,diagnostic_only=True,adopted=False,original_case_not_overwritten=True,sources=sources,raw_root=str(ROOT),payload_sha256=sha(ROOT/'stages.pt'),configuration=cfg,weight_signature=signature,
        original_failed_actual=failed['compiled_fused_actual_timed'],actual=metadata(actual_kernel),diagnostic=metadata(diagnostic_kernel),
        actual_binary_matches_original_failed=metadata(actual_kernel)['binary_sha256']==failed['compiled_fused_actual_timed']['binary_sha256'],
        integer_difference_count=int((data['native_integer']!=data['diagnostic_integer']).sum()),projection_difference_count=int((data['native_projected'].view(torch.int16)!=data['diagnostic_projected'].view(torch.int16)).sum()),
        actual_output_difference_count=int((ref_timed.view(torch.int16)!=actual_timed.view(torch.int16)).sum()),diagnostic_output_difference_count=int((ref_timed.view(torch.int16)!=data['diagnostic_output'].view(torch.int16)).sum()),guards_inputs_RNG_preserved=True)
    (ROOT/'result.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps({k:v for k,v in state.items() if k not in ('sources','weight_signature')}));signal.alarm(0)
if __name__=='__main__':main()
