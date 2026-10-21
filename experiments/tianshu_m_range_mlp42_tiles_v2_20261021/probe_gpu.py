# SPDX-License-Identifier: Apache-2.0
"""42 actual layer weight banks, generated Inductor activation, whole MLP stages."""
import argparse,hashlib,importlib.util,json,os,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
ROWS=[128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,2048,2304,2560,2816,3072,3328,3584,3840,4096]
# Configuration per current warm M is frozen in selected_plan.json before GPU.
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args();root=args.output.resolve()
    remaining=int((datetime.fromisoformat('2026-10-21T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds());assert remaining>900;signal.alarm(min(remaining,3600))
    hardware='' if args.preflight_only else subprocess.check_output(['ixsmi'],text=True)
    if not args.preflight_only:assert 'No running processes found' in hardware
    gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['CPU_only'];assert all(sha(f)==d for f,d in gate['sources'].items())
    selection=json.loads((BASE/'selected_plan.json').read_text());assert selection['component_independently_CPU_verified'] and set(map(int,selection['configuration_by_M']))==set(ROWS)
    component=json.loads((EXP/'tianshu_nn_silu_large_tiles_v2_results_20261021/GPU_GATE_32g.json').read_text());assert component['complete'] and component['CPU_only']
    assert selection['component_gate_sha256']==sha(EXP/'tianshu_nn_silu_large_tiles_v2_results_20261021/GPU_GATE_32g.json')
    assert root.is_relative_to('/workspace/benchmark-results')
    if args.preflight_only:
        assert 'torch' not in sys.modules and 'flag_gems' not in sys.modules and not root.exists()
        print(json.dumps(dict(complete=True,CPU_only=True,actual_entry_preflight=True,torch_not_imported=True,GPU_root_not_created=True,configuration_M=len(selection['configuration_by_M']))));return
    assert gate['actual_CPU_entry_preflight_passed']
    root.mkdir(exist_ok=False);(root/'hardware_before.txt').write_text(hardware)
    state=dict(complete=False,phase='starting',runner_pid=os.getpid(),component_only=True,model_executed=False,official_benchmark=False,adopted=False,sources=gate['sources'],rows=ROWS,configuration_by_M=selection['configuration_by_M'],cases=[],timing_rounds=[],bank_signatures=[])
    def save():
        p=root/'result.tmp';p.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');p.replace(root/'result.json')
    save()
    try:
        import torch,flag_gems
        from safetensors import safe_open
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        nn=load('_mlp42_nn',EXP/'tianshu_int8_gateup_prefill_grouped_20261021/kernel.py')
        fused=load('_mlp42_fused',EXP/'tianshu_nn_silu_large_tiles_v2_20261021/kernel.py')
        act=load('_mlp42_inductor_activation',BASE/'inductor_activation.py')
        current=load('_mlp42_current2048',EXP/'tianshu_prefill_gemm_tiles_20261014/experimental_mm.py')
        plan=json.loads((EXP/'tianshu_prefill_int8_service_20261021/selected_plan.json').read_text())['configurations']
        signatures={r['prefix']:r for r in json.loads((EXP/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json').read_text())}
        for f,d in gate['sources'].items():
            dest=root/'source_snapshot'/Path(f).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(f).read_bytes());assert sha(dest)==d
        def fingerprint(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
        def pack(x):
            v=x.float();s=v.abs().amax(-1).clamp_min(1.e-8)/127.;return (v/s[:,None]).round().clamp(-127,127).to(torch.int8),s
        banks=[];started=time.perf_counter()
        with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as ck:
            for index in range(42):
                prefix=f'model.layers.{index}.mlp.';w=torch.cat([ck.get_tensor(prefix+kind+'_proj.weight') for kind in ('gate','up')],0).contiguous();q,s=pack(w);down=ck.get_tensor(prefix+'down_proj.weight').contiguous();expected=signatures[prefix+'gate_up_proj']
                proof=dict(prefix=prefix+'gate_up_proj',original_BF16_sha256=fingerprint(w),packed_sha256=fingerprint(q),scales_sha256=fingerprint(s),Down_BF16_sha256=fingerprint(down))
                assert all(proof[key]==expected[key] for key in ('original_BF16_sha256','packed_sha256','scales_sha256'));state['bank_signatures'].append(proof)
                banks.append(dict(W=q.T.contiguous().cuda(),S=s.cuda(),D=down.cuda()))
                assert fingerprint(banks[-1]['W'].T.cpu().contiguous())==proof['packed_sha256']
                assert fingerprint(banks[-1]['S'].cpu())==proof['scales_sha256'] and fingerprint(banks[-1]['D'].cpu())==proof['Down_BF16_sha256']
                del w,q,s,down
        torch.cuda.synchronize();state['weight_load_seconds']=time.perf_counter()-started;state['GPU_bank_bytes']=sum(t.numel()*t.element_size() for bank in banks for t in bank.values());save()
        def activation(p,o,m):
            handle=torch._C._cuda_getCurrentRawStream(0)
            if torch.cuda.is_current_stream_capturing():state.setdefault('activation_capture_handles',[]).append(dict(M=m,handle=int(handle)))
            act.triton_poi_fused_mul_silu_slice_1.run(p,o,m*6144,stream=handle)
        def compare(x,y,atol=.02,rtol=.01):
            d=(x.float()-y.float()).abs();bad=d>atol+rtol*y.float().abs();return dict(passed=bool(torch.isfinite(x).all() and torch.isfinite(y).all() and not bad.any()),failed_elements=int(bad.sum()),max_abs=float(d.max()),atol=atol,rtol=rtol)
        def measure(g):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(3):g.replay()
            b.record();b.synchronize();return a.elapsed_time(b)/3/42
        with torch.inference_mode():
          for m in ROWS:
            CONFIG=selection['configuration_by_M'][str(m)]
            cfg=next(r['configuration'] for r in plan if r['M']>=m);ac=torch.randn(m,2048,dtype=torch.bfloat16,generator=torch.Generator().manual_seed(202610211842+m));ac[0].zero_()
            x=ac.cuda();q=torch.empty(m,2048,device='cuda',dtype=torch.int8);s=torch.empty(m,device='cuda',dtype=torch.float32)
            p=torch.empty(m,12288,device='cuda',dtype=torch.bfloat16);pi=torch.empty(m,12288,device='cuda',dtype=torch.int32);fi=torch.empty_like(pi);fp=torch.empty_like(p)
            a0=torch.empty(m,6144,device='cuda',dtype=torch.bfloat16);a1=torch.empty_like(a0);o0=torch.empty(m,2048,device='cuda',dtype=torch.bfloat16);o1=torch.empty_like(o0)
            def down(a,bank,out):
                if m==2048:current.mm(a,bank['D'],(out,out),dict(bm=256,bn=256,bk=32,splits=1,warps=16,stages=2,group_m=1))
                else:flag_gems.mm_out(a,bank['D'].T,out=out)
            def control_layer(bank):nn.quantize(x,q,s);nn.mm(q,bank['W'],s,bank['S'],p,pi,cfg);activation(p,a0,m);down(a0,bank,o0)
            def candidate_layer(bank):
                nn.quantize(x,q,s);compiled=fused.fused(q,bank['W'],s,bank['S'],a1,fi,fp,CONFIG)
                actual=state.setdefault('actual_timed_candidate_kernels',{})
                if str(m) not in actual:
                    binary=hashlib.sha256(compiled.kernel).hexdigest()
                    actual[str(m)]=dict(configuration=CONFIG,binary_sha256=binary,cache_hash=compiled.hash,registers=compiled.n_regs,spills=compiled.n_spills)
                    if str(m) in selection['anchor_candidates']:assert binary==selection['anchor_candidates'][str(m)]['binary_sha256']
                down(a1,bank,o1)
            rows=torch.tensor(sorted({0,1,m//2,m-1}));cols=torch.linspace(0,6143,16).long();full_cols=torch.cat((cols,cols+6144));down_cols=torch.linspace(0,2047,16).long();rd,cd,fc,dc=rows.cuda(),cols.cuda(),full_cols.cuda(),down_cols.cuda()
            selected={index:dict(W=banks[index]['W'].T[fc].cpu(),S=banks[index]['S'][fc].cpu(),Down_W=banks[index]['D'][dc].cpu()) for index in (0,7,20,41)}
            validation_graphs=[]
            for bank in banks:
                for _ in range(2):control_layer(bank);candidate_layer(bank)
                torch.cuda.synchronize();vgc,vgf=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
                with torch.cuda.graph(vgc):control_layer(bank)
                with torch.cuda.graph(vgf):candidate_layer(bank)
                validation_graphs.append((vgc,vgf))
            for phase in range(3):
                factor=(1.,.5,-.5)[phase];x.copy_(ac*factor);aq,asc=pack(ac*factor)
                fixture=f'M{m}_phase{phase}_fixture.pt';torch.save(dict(A_rows=(ac*factor)[rows],q_rows=aq[rows],scale_rows=asc[rows],columns=full_cols,Down_columns=down_cols,banks=selected),root/fixture)
                for index,bank in enumerate(banks):
                    a0.fill_(float('nan'));a1.fill_(float('nan'))
                    validation_graphs[index][0].replay();validation_graphs[index][1].replay();torch.cuda.synchronize()
                    graph_a0,graph_a1,graph_o0,graph_o1=a0.clone(),a1.clone(),o0.clone(),o1.clone()
                    control_layer(bank);candidate_layer(bank);torch.cuda.synchronize()
                    assert torch.equal(graph_a0.view(torch.int16),a0.view(torch.int16)) and torch.equal(graph_a1.view(torch.int16),a1.view(torch.int16)),(m,phase,index,'Graph/eager activation')
                    assert compare(graph_o0,o0)['passed'] and compare(graph_o1,o1)['passed']
                    timed_activation=compare(a1,a0,.001,.02);mlp=compare(o1,o0)
                    assert timed_activation['passed'] and mlp['passed'],(m,phase,index,timed_activation,mlp)
                    nn.mm(q,bank['W'],s,bank['S'],p,pi,cfg,return_int=True);fused.fused(q,bank['W'],s,bank['S'],a1,fi,fp,CONFIG,diagnostics=True)
                    assert torch.equal(pi,fi) and torch.equal(p.view(torch.int16),fp.view(torch.int16));assert torch.equal(q.cpu(),aq) and torch.equal(s.cpu(),asc)
                    artifact=None
                    if index in selected:
                        exact=aq[rows].to(torch.int64)@selected[index]['W'].to(torch.int64).T;actual_int=fi[rd[:,None],fc[None,:]].cpu();assert torch.equal(actual_int.to(torch.int64),exact)
                        artifact=f'M{m}_phase{phase}_layer{index}.pt';torch.save(dict(integer=actual_int,projected=fp[rd[:,None],fc[None,:]].cpu(),baseline_activation_selected=a0[rd[:,None],cd[None,:]].cpu(),fused_activation_selected=a1[rd[:,None],cd[None,:]].cpu(),baseline_activation_full_rows=a0[rd].cpu(),fused_activation_full_rows=a1[rd].cpu(),baseline_Down=o0[rd[:,None],dc[None,:]].cpu(),fused_Down=o1[rd[:,None],dc[None,:]].cpu()),root/artifact)
                    state['cases'].append(dict(M=m,phase=phase,layer=index,full_timed_activation_check=timed_activation,full_timed_MLP_check=mlp,full_INT32_and_projected_bitwise=True,full_A_quantization_exact=True,full_control_Graph_eager_activation_bitwise=True,full_candidate_Graph_eager_activation_bitwise=True,full_Graph_eager_Down_checks=True,artifact=artifact,fixture=fixture))
                save();print(json.dumps(dict(M=m,phase=phase,checked_layers=42)),flush=True)
            def control_chain():
                for bank in banks:control_layer(bank)
            def candidate_chain():
                for bank in banks:candidate_layer(bank)
            for _ in range(2):control_chain();candidate_chain()
            torch.cuda.synchronize();expected_a0,expected_a1,expected_o0,expected_o1=a0.clone(),a1.clone(),o0.clone(),o1.clone()
            gc,gf=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
            with torch.cuda.graph(gc):control_chain()
            with torch.cuda.graph(gf):candidate_chain()
            a0.fill_(float('nan'));a1.fill_(float('nan'));o0.fill_(float('nan'));o1.fill_(float('nan'))
            gc.replay();gf.replay();torch.cuda.synchronize()
            assert torch.equal(a0.view(torch.int16),expected_a0.view(torch.int16)) and torch.equal(a1.view(torch.int16),expected_a1.view(torch.int16))
            assert compare(o0,expected_o0)['passed'] and compare(o1,expected_o1)['passed']
            times={'current_MLP':[],'fused_MLP':[]};graphs={'current_MLP':gc,'fused_MLP':gf}
            for trial in range(7):
                order=['current_MLP','fused_MLP'] if trial%2==0 else ['fused_MLP','current_MLP']
                for name in order:times[name].append(measure(graphs[name]))
            state['timing_rounds'].append(dict(M=m,configuration=CONFIG,baseline_configuration=cfg,weight_banks_per_graph=42,replays_per_trial=3,times_ms_per_layer=times,medians_ms_per_layer={name:statistics.median(v) for name,v in times.items()},full_chain_poison_Graph_eager_passed=True,all_42_individual_Graph_eager_dynamic_phases_passed=True,scope='42 independent actual MLP weight banks on shared fixture X, not full transformer/model'))
            post_timing_check=compare(o1,o0);assert post_timing_check['passed'];state['timing_rounds'][-1]['last_bank_output_after_timing']=post_timing_check
            state['phase']='probing';save();print(json.dumps(state['timing_rounds'][-1]),flush=True)
            del graphs,gc,gf,validation_graphs,vgc,vgf,graph_a0,graph_a1,graph_o0,graph_o1,expected_a0,expected_a1,expected_o0,expected_o1,x,q,s,p,pi,fi,fp,a0,a1,o0,o1;torch.cuda.empty_cache()
        for index,bank in enumerate(banks):
            proof=state['bank_signatures'][index]
            assert fingerprint(bank['W'].T.cpu().contiguous())==proof['packed_sha256']
            assert fingerprint(bank['S'].cpu())==proof['scales_sha256'] and fingerprint(bank['D'].cpu())==proof['Down_BF16_sha256']
        assert all(sha(f)==d for f,d in gate['sources'].items());state.update(complete=True,phase='complete',all_42_GPU_bank_full_hashes_before_after=True);save()
    except BaseException as error:state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
