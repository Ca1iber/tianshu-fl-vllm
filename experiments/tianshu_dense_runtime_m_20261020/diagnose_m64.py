# SPDX-License-Identifier: Apache-2.0
"""Preserve original failed sweep; separately distinguish integer/BF16/alignment."""
import hashlib,importlib.util,json,os,signal,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_dense_w8a8_20261020'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    signal.alarm(240)
    root=Path('/workspace/benchmark-results/20261020_dense_runtime_m_M64_diagnostic_tianshu');assert not root.exists();root.mkdir()
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    import torch,vllm_fl
    from safetensors import safe_open
    import kernel_runtime_m as dynamic
    spec=importlib.util.spec_from_file_location('_M64_diagnostic_static',PARENT/'kernel.py');static=importlib.util.module_from_spec(spec);spec.loader.exec_module(static)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    frozen={str(p):sha(p) for p in (Path(__file__),BASE/'kernel_runtime_m.py',PARENT/'kernel.py')}
    for name,digest in frozen.items():
        dst=root/'source_snapshot'/Path(name).relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(name).read_bytes())
    result=dict(complete=False,diagnostic_only=True,not_adopted=True,original_failed_sweep_unchanged=True,runner_pid=os.getpid(),frozen_sources=frozen,cases=[])
    def save():(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def pack(a):
        s=a.float().abs().amax(1).clamp_min(1.e-8)/127.;q=torch.round(a.float()/s[:,None]).clamp(-127,127).to(torch.int8);return q,s
    generator=torch.Generator().manual_seed(202610202325)
    for m in (1,3,7,8,17,32,37,40,48,56,63,64):
        ac=torch.randn((m,2048),generator=generator,dtype=torch.bfloat16)
        if m>=3:ac[0].zero_();ac[1].zero_();ac[1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16)
    with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as f:
        wc=torch.cat([f.get_tensor(f'model.layers.0.mlp.{name}_proj.weight') for name in ('gate','up')],0).contiguous()
    qc,sc=pack(ac);wqc,wsc=pack(wc);a=ac.cuda();wq=wqc.cuda();ws=wsc.cuda();cfg=dict(bm=64,bn=256,bk=128,warps=4,stages=1)
    save()
    for prefix in (0,16,256):
        data={};stores=[]
        for name,module in [('static',static),('dynamic',dynamic)]:
            arrays={};align={}
            for field,shape,dtype in [('q',(64,2048),torch.int8),('s',(64,),torch.float32),('out',(64,12288),torch.bfloat16),('integer',(64,12288),torch.int32)]:
                count=1
                for dim in shape:count*=dim
                store=torch.full((count+prefix+256,),17,device='cuda',dtype=dtype);view=store[prefix:prefix+count].view(shape);arrays[field]=view;stores.append((store,count));align[field]=view.data_ptr()%256
            qk=module.quantize(a,arrays['q'],arrays['s']);gk=module.mm(arrays['q'],wq,arrays['s'],ws,arrays['out'],arrays['integer'],cfg,return_int=True)
            torch.cuda.synchronize();assert torch.equal(arrays['q'].cpu(),qc) and torch.equal(arrays['s'].cpu(),sc)
            data[name]=dict(arrays=arrays,alignment_mod256=align,kernel_cache_hash=gk.hash,binary_sha256=hashlib.sha256(gk.kernel).hexdigest(),n_regs=gk.n_regs)
        ref=qc.cuda().float()@wq.float().T
        order1=ref*sc.cuda()[:,None]*ws[None,:];order2=ref*(sc.cuda()[:,None]*ws[None,:])
        left,right=data['static']['arrays'],data['dynamic']['arrays']
        case=dict(prefix_elements=prefix,variants={},integer_variants_equal=torch.equal(left['integer'],right['integer']),BF16_variants_equal=torch.equal(left['out'],right['out']))
        mask=(left['integer']!=right['integer'])|(left['out']!=right['out'])
        coords=mask.nonzero()[:32];archive=[]
        for name,d in data.items():
            integers=d['arrays']['integer'];out=d['arrays']['out']
            case['variants'][name]=dict(alignment_mod256=d['alignment_mod256'],kernel_cache_hash=d['kernel_cache_hash'],binary_sha256=d['binary_sha256'],n_regs=d['n_regs'],
                full_integer_FP32_reference_exact=torch.equal(integers.float(),ref),integer_reference_mismatches=int((integers.float()!=ref).sum()),
                output_equals_FP32_order1_BF16=torch.equal(out,order1.to(torch.bfloat16)),output_equals_FP32_order2_BF16=torch.equal(out,order2.to(torch.bfloat16)),
                full_output_tolerance_passed=not bool(((out.float()-order1).abs()>(.02+.01*order1.abs())).any()),
                output_reference_max_abs=float((out.float()-order1).abs().max()))
        case['BF16_variant_mismatches']=int((left['out']!=right['out']).sum());case['bad_coordinates']=coords.cpu().tolist()
        for r,c in coords.cpu().tolist():
            exact=int(qc[r].to(torch.int64)@wqc[c].to(torch.int64))
            archive.append(dict(row=r,col=c,a_INT8=qc[r],original_a=ac[r],w_INT8=wqc[c],original_w=wc[c],ascale=sc[r],wscale=wsc[c],CPU_INT64=exact,
                static_integer=int(left['integer'][r,c]),dynamic_integer=int(right['integer'][r,c]),static_BF16=left['out'][r,c].cpu(),dynamic_BF16=right['out'][r,c].cpu()))
        case['guards_passed']=all(bool((store[:prefix]==17).all() and (store[prefix+count:]==17).all()) for store,count in stores)
        torch.save(archive,root/f'prefix{prefix}_selected.pt');result['cases'].append(case);save()
    assert all(sha(p)==v for p,v in frozen.items());result['complete']=True;save();print(json.dumps(result))
if __name__=='__main__':main()
