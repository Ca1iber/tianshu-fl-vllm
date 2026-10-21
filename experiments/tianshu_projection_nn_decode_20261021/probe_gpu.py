# SPDX-License-Identifier: Apache-2.0
"""Actual O/Down NN/NT integer and BF16 controls; complete saved integer CPU proof required."""
import argparse,gc,hashlib,importlib.util,json,os,signal,statistics,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
NT=ROOT/'tianshu_dense_w8a8_20261020'
NN=ROOT/'tianshu_int8_weight_nn_20261021'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True);parser.add_argument('--projection',choices=('o','down'),required=True);args=parser.parse_args()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200;signal.alarm(min(1800,remaining))
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
    import torch,vllm_fl,flag_gems
    from safetensors import safe_open
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;assert getattr(flag_gems,"current_work_registrar",None) is None;native=load('_nn_tiles_native',NT/'kernel.py');candidate=load('_nn_tiles_candidate',NN/'kernel_nn.py')
    gate=read(BASE/'CPU_GATE_32g.json');assert gate['complete'] and all(sha(p)==d for p,d in gate['sources'].items())
    frozen={**gate['sources'],str(BASE/'CPU_GATE_32g.json'):sha(BASE/'CPU_GATE_32g.json')}
    for p,d in frozen.items():
        dest=root/'source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes());assert sha(dest)==d
    result=dict(complete=False,phase='weights',runner_pid=os.getpid(),frozen_sources=frozen,cases=[],rounds=[],official_benchmark=False,model_executed=False,adopted=False)
    def save():
        target=root/'result.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(target)
    def fingerprint(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
    def guarded(shape,dtype,value):
        count=1
        for d in shape:count*=d
        store=torch.full((count+512,),value,dtype=dtype,device='cuda');view=store[256:256+count].view(shape);assert view.data_ptr()%256==0
        return store,view,value
    def pack(w):
        value=w.float();s=value.abs().amax(1).clamp_min(1.e-8)/127.;return torch.round(value/s[:,None]).clamp(-127,127).to(torch.int8),s
    def meta(compiled):return dict(hash=compiled.hash,binary_sha256=hashlib.sha256(compiled.kernel).hexdigest(),n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared=compiled.metadata.shared)
    def timing(graph):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
        for _ in range(40):graph.replay()
        end.record();end.synchronize();return start.elapsed_time(end)/40
    def guards(stores):
        for store,view,value in stores:assert bool((store[:256]==value).all() and (store[-256:]==value).all())
    save()
    try:
        with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as checkpoint:
            wc=checkpoint.get_tensor('model.layers.0.'+('self_attn.o_proj.weight' if args.projection=='o' else 'mlp.down_proj.weight')).contiguous()
        wqc,wsc=pack(wc);n,k=wc.shape;assert (n,k)=={'o':(2048,2048),'down':(2048,6144)}[args.projection]
        expected=gate['weight_signatures'][args.projection]
        assert fingerprint(wc)==expected['original_BF16_sha256'] and fingerprint(wqc)==expected['packed_sha256'] and fingerprint(wsc)==expected['scales_sha256']
        result['weight_signature']=expected;result['projection']=args.projection
        wsbuf=guarded((n,),torch.float32,17.);ntbuf=guarded((n,k),torch.int8,17);nnbuf=guarded((k,n),torch.int8,17)
        ws,wn,wt=wsbuf[1],ntbuf[1],nnbuf[1];result['phase']='cases';save()
        plan={(r['case'],r['M']):r['configuration'] for r in read(NT/'GPU_GATE_32g.json')['best_by_case']}
        generator=torch.Generator().manual_seed(202610212100);tiles=read(BASE/'tiles.json')
        for m in (8,64):
          ac=torch.randn((m,k),generator=generator,dtype=torch.bfloat16);ac[0].zero_();ac[1].zero_();ac[1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16)
          for tile_id,config in enumerate(tiles):
            row=dict(M=m,tile_id=tile_id,NT_configuration=plan[(args.projection,m)],NN_configuration=config,phases=[],complete=False);result['cases'].append(row);save()
            a=ac.cuda();ws.copy_(wsc);wn.copy_(wqc);wt.copy_(wqc.T.contiguous());stores=[wsbuf,ntbuf,nnbuf];variants={}
            try:
                for name,module,w,cfg in [('NT',native,wn,plan[(args.projection,m)]),('NN',candidate,wt,config)]:
                    data={}
                    for key,shape,dtype,value in [('q',(m,k),torch.int8,17),('s',(m,),torch.float32,17.),('o',(m,n),torch.bfloat16,17.),('i',(m,n),torch.int32,-777)]:
                        buffer=guarded(shape,dtype,value);stores.append(buffer);data[key]=buffer[1]
                    for _ in range(3):module.quantize(a,data['q'],data['s']);compiled=module.mm(data['q'],w,data['s'],ws,data['o'],data['i'],cfg)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):module.quantize(a,data['q'],data['s']);module.mm(data['q'],w,data['s'],ws,data['o'],data['i'],cfg)
                    variants[name]=dict(data=data,module=module,w=w,cfg=cfg,graph=graph,compiled=meta(compiled))
            except Exception as error:
                if type(error).__name__ not in ('OutOfResources','CompilationError'):raise
                row.update(compile_failed=True,exception_type=type(error).__name__,exception=str(error));save();print(json.dumps({k:v for k,v in row.items() if k!='phases'}),flush=True);variants.clear();gc.collect();torch.cuda.empty_cache();continue
            row['compiled']={name:v['compiled'] for name,v in variants.items()}
            bfweight=wc.cuda();bfout=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
            for _ in range(2):flag_gems.mm_out(a,bfweight.T,out=bfout)
            bfgraph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(bfgraph):flag_gems.mm_out(a,bfweight.T,out=bfout)
            columns=torch.linspace(0,n-1,32).long();rows=torch.linspace(0,m-1,min(m,8)).long();cd,rd=columns.cuda(),rows.cuda()
            for phase in range(3):
                actual=ac if phase==0 else ac*(.5 if phase==1 else -.5);sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.
                a.copy_(actual);bfweight.copy_(wc*sign*factor);wn.copy_(wqc*sign);wt.copy_((wqc*sign).T.contiguous());ws.copy_(wsc*factor);torch.cuda.synchronize();assert torch.equal(wt.T,wn)
                rng_cpu=torch.get_rng_state().clone();rng_gpu=torch.cuda.get_rng_state().clone();aq,asc=pack(actual)
                for value in variants.values():
                    value['graph'].replay();d=value['data'];value['module'].mm(d['q'],value['w'],d['s'],ws,d['o'],d['i'],value['cfg'],return_int=True)
                    assert torch.equal(d['q'].cpu(),aq) and torch.equal(d['s'].cpu(),asc)
                l,r=variants['NT']['data'],variants['NN']['data'];assert torch.equal(l['i'],r['i']) and torch.equal(l['o'].view(torch.int16),r['o'].view(torch.int16)) and bool(torch.isfinite(l['o']).all())
                bfgraph.replay();torch.cuda.synchronize();bfref=(a.float()@bfweight.float().T).to(torch.bfloat16);bferror=(bfout.float()-bfref.float()).abs();bfvalid=bool(torch.isfinite(bfout).all() and not (bferror>(.02+.01*bfref.float().abs())).any())
                if not bfvalid:torch.save(dict(A=a.cpu(),W=bfweight.cpu(),actual_native_BF16=bfout.cpu(),GPU_FP32_to_BF16_reference=bfref.cpu()),root/f'failed_native_M{m}_tile{tile_id}_phase{phase}.pt')
                exact=aq[rows].to(torch.int64)@(wqc[columns]*sign).to(torch.int64).T;actual_integer=l['i'][rd[:,None],cd[None,:]].cpu()
                assert torch.equal(actual_integer.to(torch.int64),exact)
                assert torch.equal(a.cpu(),actual) and torch.equal(wn.cpu(),wqc*sign) and torch.equal(wt.cpu(),(wqc*sign).T.contiguous()) and torch.equal(ws.cpu(),wsc*factor)
                assert torch.equal(torch.get_rng_state(),rng_cpu) and torch.equal(torch.cuda.get_rng_state(),rng_gpu);guards(stores)
                filename=f'M{m}_tile{tile_id}_phase{phase}.pt'
                torch.save(dict(full_GPU_integer=variants['NN']['data']['i'].cpu(),full_GPU_BF16=variants['NN']['data']['o'].cpu(),native_BF16_selected=bfout[rd[:,None],cd[None,:]].cpu(),a=actual,q=variants['NN']['data']['q'].cpu(),ascale=variants['NN']['data']['s'].cpu(),rows=rows,columns=columns,original_W_rows=wc[columns],actual_GPU_integer=actual_integer,actual_GPU_BF16=variants['NN']['data']['o'][rd[:,None],cd[None,:]].cpu(),phase=phase),root/filename)
                row['phases'].append(dict(phase=phase,native_BF16_full_valid=bfvalid,native_BF16_max_abs=float(bferror.max()),full_INT32_bitwise=True,full_BF16_bitwise=True,all_A_quantization=True,CPU_selected_INT64=True,full_weights_immutable=True,guards=True,RNG_unchanged=True,artifact=filename))
            for index in range(7):
                order=['NT','NN'] if index%2==0 else ['NN','NT'];order.append('BF16') if index%2==0 else order.insert(0,'BF16');times={name:timing(bfgraph if name=='BF16' else variants[name]['graph']) for name in order}
                result['rounds'].append(dict(M=m,tile_id=tile_id,round=index,order=order,replays=40,**times))
            guards(stores);row.update(complete=True,timing_guards_passed=True);save();print(json.dumps(dict(M=m,tile_id=tile_id,complete=True)),flush=True)
            variants.clear();del a,l,r,variants,bfgraph,bfweight,bfout,bfref,bferror;gc.collect();torch.cuda.empty_cache()
        assert all(sha(p)==d for p,d in frozen.items());result.update(complete=True,phase='complete');save()
    except BaseException as error:result.update(phase='failed',failure=repr(error));save();raise
    finally:(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
