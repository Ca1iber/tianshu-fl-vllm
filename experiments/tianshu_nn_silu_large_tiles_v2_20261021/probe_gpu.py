# SPDX-License-Identifier: Apache-2.0
"""Broaden two-dot BK/BN/warp choices against the actual existing M64 fusion."""
import argparse,gc,hashlib,importlib.util,json,signal,statistics,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
ROWS=[8,64,128,384,512,1024,2048,4096]
TILES=[(64,128,64,4),(64,128,64,8),(64,256,64,4),(64,256,64,8),(128,128,64,4),(128,128,64,8),(128,256,64,8),(128,256,64,16),(128,256,32,8),(128,256,32,16),(128,512,32,16),(128,512,64,16),(256,128,32,8),(256,128,32,16),(256,256,32,16)]
CONFIGS=[dict(bm=bm,bn=bn,bk=bk,warps=w,stages=1,mode='two_dot',group_m=g) for bm,bn,bk,w in TILES for g in (1,8)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output.resolve()
    remaining=int((datetime.fromisoformat('2026-10-21T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert remaining>1200;signal.alarm(min(remaining,5400))
    hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw
    freeze_gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert freeze_gate['complete'] and freeze_gate['CPU_only'];assert all(sha(f)==d for f,d in freeze_gate['sources'].items())
    assert root.is_relative_to('/workspace/benchmark-results');root.mkdir(exist_ok=False);(root/'hardware_before.txt').write_text(hw)
    result=dict(complete=False,phase='starting',component_only=True,model_executed=False,adopted=False,sources=freeze_gate['sources'],rows=ROWS,configurations=CONFIGS,cases=[])
    def save():
        p=root/'result.tmp';p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');p.replace(root/'result.json')
    save()
    try:
        import torch
        from safetensors import safe_open
        torch.set_num_threads(8)
        nn=load('_large_tiles_nn',EXP/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py')
        old_fused=load('_large_tiles_original_fused',EXP/'tianshu_nn_silu_two_dot_20261021/kernel.py')
        original_gate=json.loads((EXP/'tianshu_nn_silu_two_dot_20261021/GPU_GATE_32g.json').read_text());assert original_gate['complete']
        original_binary=next(row['actual_timed_compiled']['binary_sha256'] for row in original_gate['best_by_shape'] if row['M']==64 and not row['strided'])
        candidate=load('_large_tiles_candidate',BASE/'kernel.py')
        activation=load('_large_tiles_inductor_activation',EXP/'tianshu_m_range_mlp42_v2_20261021/inductor_activation.py')
        for f,d in freeze_gate['sources'].items():
            dest=root/'source_snapshot'/Path(f).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(f).read_bytes());assert sha(dest)==d
        def pack(t):
            v=t.float();s=v.abs().amax(-1).clamp_min(1.e-8)/127.;return (v/s[:,None]).round().clamp(-127,127).to(torch.int8),s
        def fingerprint(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
        def check(x,y,atol=.001,rtol=.02):
            delta=(x.float()-y.float()).abs();bad=delta>atol+rtol*y.float().abs();return dict(passed=bool(torch.isfinite(x).all() and torch.isfinite(y).all() and not bad.any()),bad=int(bad.sum()),max_abs=float(delta.max()),atol=atol,rtol=rtol)
        def meta(k):return dict(hash=k.hash,binary_sha256=hashlib.sha256(k.kernel).hexdigest(),registers=k.n_regs,spills=k.n_spills,shared=k.metadata.shared)
        def measure(g):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(30):g.replay()
            b.record();b.synchronize();return a.elapsed_time(b)/30
        def capture(fn):
            for _ in range(2):fn()
            torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):fn()
            return g
        with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as f:
            wc=torch.cat([f.get_tensor('model.layers.0.mlp.'+part+'_proj.weight') for part in ('gate','up')],0).contiguous()
        wqc,wsc=pack(wc);wt=wqc.T.contiguous().cuda();ws=wsc.cuda();signature=next(r for r in json.loads((EXP/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text()) if r['prefix']=='model.layers.0.mlp.gate_up_proj')
        assert fingerprint(wc)==signature['original_BF16_sha256'] and fingerprint(wqc)==signature['packed_sha256'] and fingerprint(wsc)==signature['scales_sha256'];result['weight_signature']=signature
        plan=json.loads((EXP/'tianshu_prefill_int8_service_20261021/selected_plan.json').read_text())['configurations']
        old_config=dict(bm=64,bn=64,bk=128,warps=4,stages=1,mode='two_dot',group_m=1)
        with torch.inference_mode():
          for m in ROWS:
            a_cpu=torch.randn(m,2048,dtype=torch.bfloat16,generator=torch.Generator().manual_seed(202610212200+m));a_cpu[0].zero_();a_cpu[1].zero_();a_cpu[1,:9]=torch.tensor([0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127],dtype=torch.bfloat16)
            a=a_cpu.cuda();q=torch.empty(m,2048,device='cuda',dtype=torch.int8);s=torch.empty(m,device='cuda',dtype=torch.float32)
            proj=torch.empty(m,12288,device='cuda',dtype=torch.bfloat16);integer=torch.empty(m,12288,device='cuda',dtype=torch.int32);cproj=torch.empty_like(proj);cint=torch.empty_like(integer)
            ref_out=torch.empty(m,6144,device='cuda',dtype=torch.bfloat16);current_out=torch.empty_like(ref_out);out=torch.empty_like(ref_out)
            base_cfg=(dict(bm=32 if m==8 else 64,bn=256,bk=128,warps=4,stages=1,group_m=1) if m in (8,64) else next(r['configuration'] for r in plan if r['M']>=m))
            def reference():nn.quantize(a,q,s);nn.mm(q,wt,s,ws,proj,integer,base_cfg);activation.triton_poi_fused_mul_silu_slice_1.run(proj,ref_out,m*6144,stream=torch._C._cuda_getCurrentRawStream(0))
            def current():
                if m==64:nn.quantize(a,q,s);return old_fused.fused(q,wt,s,ws,current_out,cint,cproj,old_config)
                else:reference()
            rows=torch.linspace(0,m-1,min(m,8)).long();cols=torch.linspace(0,6143,16).long();full_cols=torch.cat((cols,cols+6144));rd,cd,fc=rows.cuda(),cols.cuda(),full_cols.cuda()
            for phase in range(3):
                actual=a_cpu if phase==0 else a_cpu*(.5 if phase==1 else -.5);sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.;aq,asc=pack(actual)
                torch.save(dict(A=actual,q=aq,scale=asc,rows=rows,columns=full_cols,W=wqc[full_cols]*sign,W_scale=wsc[full_cols]*factor),root/f'fixture_M{m}_phase{phase}.pt')
            for ci,cfg in enumerate(CONFIGS):
                case=dict(M=m,config_id=ci,configuration=cfg,phases=[],complete=False,qualified=False);result['cases'].append(case);result['phase']='probing';save()
                a.copy_(a_cpu);wt.copy_(wqc.T.contiguous());ws.copy_(wsc)
                def invoke():nn.quantize(a,q,s);return candidate.fused(q,wt,s,ws,out,cint,cproj,cfg)
                try:
                    if m==64:
                        actual_current=current();case['actual_current_compiled']=meta(actual_current);assert case['actual_current_compiled']['binary_sha256']==original_binary
                    compiled=invoke();case['compiled']=meta(compiled);g_candidate=capture(invoke);g_ref=capture(reference);g_current=capture(current)
                except Exception as error:
                    if type(error).__name__ not in ('OutOfResources','CompilationError'):raise
                    case.update(compile_failed=True,compile_error=repr(error),complete=True);save();gc.collect();torch.cuda.empty_cache();continue
                for phase in range(3):
                    actual=a_cpu if phase==0 else a_cpu*(.5 if phase==1 else -.5);sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.;a.copy_(actual);wt.copy_((wqc*sign).T.contiguous());ws.copy_(wsc*factor)
                    g_ref.replay();g_current.replay();g_candidate.replay();torch.cuda.synchronize();timed=out.cpu().clone();graph_reference=ref_out.clone();reference();torch.cuda.synchronize();assert torch.equal(graph_reference.view(torch.int16),ref_out.view(torch.int16)),(m,ci,phase,'reference Graph/eager stale');ref_check=check(out,ref_out);current_check=check(out,current_out) if m==64 else ref_check
                    m64_bitwise=(bool(torch.equal(out.view(torch.int16),current_out.view(torch.int16))) if m==64 else None)
                    nn.mm(q,wt,s,ws,proj,integer,base_cfg,return_int=True);candidate.fused(q,wt,s,ws,out,cint,cproj,cfg,diagnostics=True)
                    integer_exact=bool(torch.equal(integer,cint));projection_exact=bool(torch.equal(proj.view(torch.int16),cproj.view(torch.int16)));aq,asc=pack(actual);quant_exact=bool(torch.equal(q.cpu(),aq) and torch.equal(s.cpu(),asc))
                    dot=aq[rows].to(torch.int64)@(wqc[full_cols]*sign).to(torch.int64).T;actual_int=cint[rd[:,None],fc[None,:]].cpu();cpu_exact=bool(torch.equal(actual_int.to(torch.int64),dot))
                    passed=bool(ref_check['passed'] and current_check['passed'] and integer_exact and projection_exact and quant_exact and cpu_exact and (m!=64 or m64_bitwise))
                    artifact=f'M{m}_cfg{ci}_phase{phase}.pt';torch.save(dict(integer=actual_int,projected=cproj[rd[:,None],fc[None,:]].cpu(),output=timed[rows[:,None],cols[None,:]],reference_output=ref_out[rd[:,None],cd[None,:]].cpu(),current_output=current_out[rd[:,None],cd[None,:]].cpu() if m==64 else None),root/artifact)
                    case['phases'].append(dict(phase=phase,passed=passed,reference_Graph_eager_BF16_bitwise=True,actual_timed_activation=ref_check,M64_current_check=current_check,M64_timed_bitwise=m64_bitwise,full_INT32_exact=integer_exact,full_projected_BF16_exact=projection_exact,full_quantization_exact=quant_exact,CPU_INT64_exact=cpu_exact,artifact=artifact))
                    if not passed:
                        failure_path=root/f'M{m}_first_failed_full.pt'
                        if not failure_path.exists():torch.save(dict(config_id=ci,phase=phase,actual_timed_output=timed,reference=ref_out.cpu(),integer=cint.cpu(),reference_integer=integer.cpu()),failure_path)
                        break
                if len(case['phases'])==3 and all(ph['passed'] for ph in case['phases']):
                    times={'current':[],'reference_NN_Inductor':[],'candidate':[]};graphs={'current':g_current,'reference_NN_Inductor':g_ref,'candidate':g_candidate}
                    for trial in range(7):
                        order=list(graphs);order=order[trial%3:]+order[:trial%3]
                        if trial%2:order.reverse()
                        for name in order:times[name].append(measure(graphs[name]))
                    g_ref.replay();g_current.replay();g_candidate.replay();torch.cuda.synchronize();after=check(out,current_out if m==64 else ref_out)
                    assert after['passed'];case.update(qualified=True,times_ms=times,medians_ms={name:statistics.median(v) for name,v in times.items()},after_timing=after)
                case['complete']=True;save();print(json.dumps(dict(M=m,config_id=ci,qualified=case['qualified'],configuration=cfg,medians_ms=case.get('medians_ms'),compiled=case['compiled'])),flush=True)
                del g_candidate,g_ref,g_current
            del a,q,s,proj,integer,cproj,cint,ref_out,current_out,out;gc.collect();torch.cuda.empty_cache()
        assert all(sha(f)==d for f,d in freeze_gate['sources'].items());result.update(complete=True,phase='complete');save()
    except BaseException as error:result.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
