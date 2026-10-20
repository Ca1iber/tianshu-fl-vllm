# SPDX-License-Identifier: Apache-2.0
"""Real-weight NT vs static-transposed NN INT8, same full A quantize and tile."""
import argparse,gc,hashlib,importlib.util,json,os,signal,statistics,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_dense_w8a8_20261020'
MS=(1,3,7,8,17,32,37,40,48,56,63,64)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200;signal.alarm(min(1800,remaining))
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
    import torch,vllm_fl
    from safetensors import safe_open
    import kernel_nn as dynamic
    spec=importlib.util.spec_from_file_location('_qualified_NT_dense_kernel',PARENT/'kernel.py');static=importlib.util.module_from_spec(spec);spec.loader.exec_module(static)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'] and all(sha(p)==v for p,v in gate['sources'].items())
    frozen={**gate['sources'],str(BASE/'CPU_GATE_32g.json'):sha(BASE/'CPU_GATE_32g.json')}
    for name,digest in frozen.items():
        target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());assert sha(target)==digest
    result=dict(complete=False,phase='weights',runner_pid=os.getpid(),frozen_sources=frozen,cases=[],timing_rounds=[],weights={},official_benchmark=False,model_executed=False,adopted=False,
        scope='Actual checkpoint banks. Same tile and A quantization, NT W[N,K] vs NN W[K,N]. Static weight packing/transpose/H2D excluded; complete A quantize+INT8 GEMM included.',
        numerical_scope='Full device INT32/BF16 equality, exact FP32 integer reference only with actual absolute sum <2^24, CPU all A/selected W requantization+INT64. Saved PT outputs are selected coordinates only.')
    def save():
        p=root/'result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(p)
    def tensor_sha(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
    def pack(w):
        q=torch.empty(w.shape,dtype=torch.int8);s=torch.empty(w.shape[0],dtype=torch.float32)
        for lo in range(0,len(w),8192):
            block=w[lo:lo+8192].float();scale=block.abs().amax(1).clamp_min(1.e-8)/127.
            q[lo:lo+len(block)]=torch.round(block/scale[:,None]).clamp(-127,127).to(torch.int8);s[lo:lo+len(block)]=scale
        return q,s
    def guarded(shape,dtype,value):
        count=1
        for dim in shape:count*=dim
        store=torch.full((count+512,),value,device='cuda',dtype=dtype);view=store[256:256+count].view(shape)
        assert view.data_ptr()%256==0;return store,view
    def metadata(kernel):return dict(cache_hash=kernel.hash,binary_sha256=hashlib.sha256(kernel.kernel).hexdigest(),name=kernel.name,n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None),shared=getattr(kernel.metadata,'shared',None))
    def measure(graph):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
        for _ in range(20):graph.replay()
        end.record();end.synchronize();return start.elapsed_time(end)/20
    signature={r['prefix']:r for r in read(BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json')}
    plan={(r['case'],r['M']):r['configuration'] for r in read(PARENT/'MODEL_INTEGRATION_PLAN_32g.json')['selected_configurations']}
    generator=torch.Generator().manual_seed(202610211517)
    model=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');save()
    try:
      with safe_open(str(model),framework='pt',device='cpu') as checkpoint:
       for role,prefix,keys in [('gateup','model.layers.0.mlp.gate_up_proj',['model.layers.0.mlp.gate_proj.weight','model.layers.0.mlp.up_proj.weight']),('lm_head','lm_head',['lm_head.weight'])]:
        start=time.perf_counter();wc=torch.cat([checkpoint.get_tensor(k) for k in keys],0).contiguous();wqc_full,wsc_full=pack(wc);full_n,full_k=wc.shape;expected=signature[prefix]
        proof=dict(original_BF16_sha256=tensor_sha(wc),packed_sha256=tensor_sha(wqc_full),scales_sha256=tensor_sha(wsc_full),shape=[full_n,full_k])
        assert all(proof[key]==expected[key] for key in proof);result['weights'][role]=dict(proof,static_weight_read_pack_seconds=time.perf_counter()-start)
        definitions=[(m,False,False) for m in MS]+[(37,True,False),(17,True,True)]
        for m,strided,ragged in definitions:
          wqc=wqc_full if not ragged else wqc_full[:130,:259].contiguous();wsc=wsc_full if not ragged else wsc_full[:130];n,k=wqc.shape
          configuration=plan[(role,8 if m<=32 else 64)]
          case=dict(role=role,M=m,N=n,K=k,strided=strided,ragged=ragged,allocated_M=m,configuration=configuration,phases=[],phase_files=[]);result['cases'].append(case);save()
          ac=torch.randn((m,k),generator=generator,dtype=torch.bfloat16)
          if m>=3:ac[0].zero_();ac[1].zero_();ac[1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16)
          astore=torch.full((m,k+(7 if strided else 0)),17.,device='cuda',dtype=torch.bfloat16);a=astore[:,:k];a.copy_(ac)
          nt_store,wq=guarded((n,k),torch.int8,17);nn_store,wqt=guarded((k,n),torch.int8,17);ws_store,ws=guarded((n,),torch.float32,17.)
          prep=time.perf_counter();wt_cpu=wqc.T.contiguous();wq.copy_(wqc);wqt.copy_(wt_cpu);ws.copy_(wsc);torch.cuda.synchronize();case['initial_static_layout_H2D_seconds_excluded']=time.perf_counter()-prep
          variants={}
          for name,module,weight in [('static',static,wq),('dynamic',dynamic,wqt)]:
            arrays={};buffers={}
            for field,shape,dtype,value in [('q',(m,k),torch.int8,17),('s',(m,),torch.float32,17.),('out',(m,n),torch.bfloat16,17.),('integer',(m,n),torch.int32,-777)]:
              store,view=guarded(shape,dtype,value);arrays[field]=view;buffers[field]=(store,value)
            for _ in range(3):qkernel=module.quantize(a,arrays['q'],arrays['s']);gkernel=module.mm(arrays['q'],weight,arrays['s'],ws,arrays['out'],arrays['integer'],configuration)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):module.quantize(a,arrays['q'],arrays['s']);module.mm(arrays['q'],weight,arrays['s'],ws,arrays['out'],arrays['integer'],configuration)
            variants[name]=dict(module=module,weight=weight,arrays=arrays,buffers=buffers,graph=graph,quantize=metadata(qkernel),gemm=metadata(gkernel))
          case['compiled']={name:{key:data[key] for key in ('quantize','gemm')} for name,data in variants.items()}
          columns=torch.linspace(0,n-1,32).long();rows=torch.linspace(0,m-1,min(m,8)).long();cd,rd=columns.cuda(),rows.cuda()
          for phase in range(3):
            actual_a=ac if phase==0 else ac*(.5 if phase==1 else -.5);sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.
            prep=time.perf_counter();actual_w=wqc*sign;actual_wt=actual_w.T.contiguous();a.copy_(actual_a);wq.copy_(actual_w);wqt.copy_(actual_wt);ws.copy_(wsc*factor);torch.cuda.synchronize();prep_s=time.perf_counter()-prep
            assert wq.is_contiguous() and wqt.is_contiguous() and torch.equal(wqt.T,wq)
            before=astore.clone();before_w=wq.clone();before_wt=wqt.clone();before_s=ws.clone();cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();aq,asc=pack(actual_a)
            for name,data in variants.items():
              data['graph'].replay();d=data['arrays'];data['module'].mm(d['q'],data['weight'],d['s'],ws,d['out'],d['integer'],configuration,return_int=True)
              assert torch.equal(d['q'].cpu(),aq) and torch.equal(d['s'].cpu(),asc)
            bound=variants['static']['arrays']['q'].float().abs()@wq.float().abs().T;max_abs=float(bound.max());assert max_abs<2**24
            reference=variants['static']['arrays']['q'].float()@wq.float().T;l=variants['static']['arrays'];r=variants['dynamic']['arrays']
            assert torch.equal(l['integer'],r['integer']) and torch.equal(l['integer'].float(),reference) and torch.equal(l['out'].view(torch.int16),r['out'].view(torch.int16))
            assert bool(torch.isfinite(l['out']).all());expected_out=reference*l['s'][:,None]*ws[None,:]
            assert not bool(((l['out'].float()-expected_out).abs()>(.02+.01*expected_out.abs())).any())
            cpu_dot=aq[rows].to(torch.int64)@actual_w[columns].to(torch.int64).T;assert torch.equal(l['integer'][rd[:,None],cd[None,:]].cpu().to(torch.int64),cpu_dot)
            assert torch.equal(astore,before) and torch.equal(wq,before_w) and torch.equal(wqt,before_wt) and torch.equal(ws,before_s)
            assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
            for data in variants.values():
              for store,value in data['buffers'].values():assert bool((store[:256]==value).all() and (store[-256:]==value).all())
            for store,value in ((nt_store,17),(nn_store,17),(ws_store,17.)):assert bool((store[:256]==value).all() and (store[-256:]==value).all())
            filename=f'{role}_M{m}_strided{int(strided)}_ragged{int(ragged)}_phase{phase}.pt'
            torch.save(dict(phase=phase,K=k,a=actual_a,original_W_rows=wc[columns].clone(),actual_A_INT8=aq,actual_A_scale=asc,actual_W_INT8=actual_w[columns],actual_W_scale=wsc[columns]*factor,
                rows=rows,columns=columns,integer=cpu_dot,static=l['out'][rd[:,None],cd[None,:]].cpu(),dynamic=r['out'][rd[:,None],cd[None,:]].cpu()),root/filename)
            case['phase_files'].append(filename);case['phases'].append(dict(phase=phase,full_INT32_exact=True,full_BF16_variants_bitwise_equal=True,full_W_NN_transpose_bytes_verified=True,
                full_quantized_output_passed=True,all_A_quantization_exact=True,CPU_selected_INT64_exact=True,actual_max_sum_abs_integer_products=max_abs,guards_passed=True,RNG_unchanged=True,inputs_unchanged=True,static_layout_phase_update_seconds_excluded=prep_s))
            del before,before_w,before_wt,before_s,bound,reference,expected_out
          for index in range(7):
            order=['static','dynamic'] if index%2==0 else ['dynamic','static'];times={name:measure(variants[name]['graph']) for name in order}
            result['timing_rounds'].append(dict(role=role,M=m,strided=strided,ragged=ragged,round=index,replays=20,order=order,static_ms=times['static'],dynamic_ms=times['dynamic']))
          for data in variants.values():
            for store,value in data['buffers'].values():assert bool((store[:256]==value).all() and (store[-256:]==value).all())
          for store,value in ((nt_store,17),(nn_store,17),(ws_store,17.)):assert bool((store[:256]==value).all() and (store[-256:]==value).all())
          assert torch.equal(astore[:,:k],actual_a.cuda()) and bool((astore[:,k:]==17).all()) and torch.equal(wqt.T,wq)
          case['timing_guards_passed']=True;save();print(json.dumps(dict(role=role,M=m,strided=strided,ragged=ragged,complete=True)),flush=True)
          variants.clear();del l,r,variants,astore,a,ac,wq,wqt,ws,nt_store,nn_store,ws_store;gc.collect();torch.cuda.empty_cache()
        del wc,wqc_full,wsc_full;gc.collect();torch.cuda.empty_cache()
      assert len(result['cases'])==28 and all(sha(p)==v for p,v in frozen.items());result.update(complete=True,phase='complete',model_quality_not_evaluated=True);save()
    except BaseException as error:result.update(phase='failed',failure=repr(error));save();raise
    finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
