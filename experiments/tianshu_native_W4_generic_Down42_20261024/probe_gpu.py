# SPDX-License-Identifier: Apache-2.0
"""Two native W4 backends, same quantized matrices and forward-only timing."""
import os,signal,subprocess,statistics,traceback
from datetime import datetime,timezone
from common import *

def main():
    remaining=int((datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds())
    assert remaining>1200;signal.alarm(min(1800,remaining));GPU.mkdir(exist_ok=False)
    state=dict(complete=False,phase='initializing',pid=os.getpid(),numeric=[],rounds=[],native_packs=[],
        prepacking_inside_timing=False,model_executed=False,official_benchmark=False,adopted=False)
    def save():
        path=GPU/'result.tmp';path.write_text(json.dumps(state,indent=2)+'\n');path.replace(GPU/'result.json')
    save()
    try:
        hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw and 'Iluvatar BI-V150' in hw
        (GPU/'hardware_before.txt').write_text(hw)
        gate=json.loads((BASE/'CPU_GATE_32g.json').read_text())
        assert gate['complete'] and gate['all42_same_full_dequantized_BF16_words']
        assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['libraries'].items())
        assert all(sha(p)==h for p,h in gate['files'].items())
        import torch,vllm_fl
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        state['actual_schemas']=dict(
            converter=str(torch.ops.aten._convert_weight_to_int4pack.default._schema),
            generic=str(torch.ops.aten._weight_int4pack_mm.default._schema),
            dotsaddz=str(torch.ops.aten._weight_int4pack_mm_dotsaddz.default._schema))
        guarded=[];immutable=[]
        def track(t,name,expected=None):
            actual=ahash(t.contiguous().view(torch.uint8).cpu().numpy())
            if expected is not None:assert actual==expected,name
            immutable.append(dict(name=name,tensor=t,hash=actual,version=int(t._version)))
            return t
        def tensor(array,dtype,name):
            array=np.ascontiguousarray(array);cpu=torch.from_numpy(array.copy())
            if array.dtype==np.uint16:cpu=cpu.view(dtype)
            raw=torch.full((array.nbytes+8192,),113,dtype=torch.uint8,device='cuda')
            t=raw[4096:-4096].view(cpu.dtype).reshape(cpu.shape);t.copy_(cpu);guarded.append(raw)
            return track(t,name,ahash(array))
        state['phase']='prepacking';save();banks=[]
        for entry in gate['banks']:
            layer=entry['layer'];a=np.load(entry['file'],allow_pickle=False)
            raw,params,proof=generic_arrays(a['W'],a['S'],a['Z']);assert proof==entry['generic']
            w=tensor(a['W'],None,f'W{layer}');s=tensor(a['S'],torch.bfloat16,f'S{layer}');z=tensor(a['Z'],torch.bfloat16,f'Z{layer}')
            u=tensor(raw,None,f'U8{layer}');p=tensor(params,torch.bfloat16,f'P{layer}')
            packed=torch.ops.aten._convert_weight_to_int4pack.default(u,8);torch.cuda.synchronize()
            assert packed.dtype==torch.int32 and list(packed.shape)==proof['expected_native_pack_shape']
            track(packed,f'CONVERTED{layer}')
            record=dict(layer=layer,shape=list(packed.shape),stride=list(packed.stride()),dtype=str(packed.dtype),
                bytes=packed.numel()*packed.element_size(),sha256=immutable[-1]['hash'],params_sha256=ahash(params),
                params_bytes=p.numel()*p.element_size(),raw_input_sha256=ahash(raw),
                raw_input_preserved_for_component_verification_only=True,converter_executed_before_timing=True)
            state['native_packs'].append(record)
            h=parent.hwords(0,layer)
            banks.append(dict(layer=layer,x=tensor(h,torch.bfloat16,f'X{layer}'),w=w,s=s,z=z,generic=packed,p=p))
            if layer%8==0:save()
        assert len(banks)==42
        operations={
            VARIANTS[0]:lambda:[torch.ops.aten._weight_int4pack_mm_dotsaddz.default(b['x'],b['w'],128,b['s'],b['z'],size_n=2048) for b in banks],
            VARIANTS[1]:lambda:[torch.ops.aten._weight_int4pack_mm.default(b['x'],b['generic'],128,b['p']) for b in banks]}
        graphs={};outputs={}
        state['phase']='warming';save()
        for name,fn in operations.items():
            for _ in range(3):fn()
            torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=fn()
            assert len(out)==42 and all(t.dtype==torch.bfloat16 and tuple(t.shape)==(64,2048) for t in out)
            graphs[name]=g;outputs[name]=out
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        def phase_inputs(phase):
            for b in banks:
                h=parent.hwords(phase,b['layer']);b['x'].copy_(torch.from_numpy(h).view(torch.bfloat16))
                entry=next(e for e in immutable if e['name']==f"X{b['layer']}")
                entry.update(hash=ahash(h),version=int(b['x']._version))
        def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
        refs={(r['layer'],r['phase']):r for r in gate['references']}
        for phase in PHASES:
            phase_inputs(phase);state['phase']=f'numeric{phase}';save()
            for name,fn in operations.items():
                eager=fn();torch.cuda.synchronize();ew=[words(t) for t in eager]
                repeated=fn();torch.cuda.synchronize();rw=[words(t) for t in repeated]
                for t in outputs[name]:t.fill_(float('nan'))
                graphs[name].replay();torch.cuda.synchronize();gw=[words(t) for t in outputs[name]]
                path=GPU/f'{name}_phase{phase}.npz'
                np.savez(path,**{f'L{layer}':raw for layer,raw in enumerate(gw)})
                extras={};relations=[];checks=[]
                for layer in range(42):
                    repeats=int(np.count_nonzero(ew[layer]!=rw[layer]))
                    graphdiff=int(np.count_nonzero(ew[layer]!=gw[layer]))
                    repeatcheck=metrics(f32(rw[layer]),f32(ew[layer]),'BF16')
                    graphcheck=metrics(f32(gw[layer]),f32(ew[layer]),'BF16')
                    relations.append(dict(layer=layer,repeat_word_differences=repeats,Graph_word_differences=graphdiff,
                        repeat_vs_eager=repeatcheck,Graph_vs_eager=graphcheck))
                    if layer in LAYERS or repeats or graphdiff:
                        extras[f'E{layer}']=ew[layer];extras[f'R{layer}']=rw[layer]
                    if layer not in LAYERS:continue
                    reference=np.load(refs[(layer,phase)]['file'],allow_pickle=False)['quantized']
                    for execution,raw in [('eager',ew[layer]),('repeat',rw[layer]),('Graph',gw[layer])]:
                        check=metrics(f32(raw),reference,'BF16')
                        checks.append(dict(layer=layer,execution=execution,**check))
                extra=GPU/f'{name}_phase{phase}_eager_repeat.npz';np.savez(extra,**extras)
                finite=all(np.isfinite(f32(raw)).all() for raw in ew+rw+gw)
                row=dict(phase=phase,variant=name,file=str(path),sha256=sha(path),
                    extra_file=str(extra),extra_sha256=sha(extra),relations=relations,selected_FP64=checks,
                    all42_finite=bool(finite),all42_repeat_numeric_passed=all(r['repeat_vs_eager']['passed'] for r in relations),
                    all42_Graph_numeric_passed=all(r['Graph_vs_eager']['passed'] for r in relations),
                    all42_repeat_word_exact=all(r['repeat_word_differences']==0 for r in relations),
                    all42_Graph_word_exact=all(r['Graph_word_differences']==0 for r in relations))
                state['numeric'].append(row);save()
                assert finite and row['all42_repeat_numeric_passed'] and row['all42_Graph_numeric_passed'] and all(c['passed'] for c in checks),row
        phase_inputs(0)
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        state['phase']='timing';save()
        for rid in range(9):
            order=VARIANTS if rid%2==0 else list(reversed(VARIANTS));times={}
            for name in order:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
                for _ in range(20):graphs[name].replay()
                end.record();end.synchronize();times[name]=start.elapsed_time(end)/20
            row=dict(round=rid,replays=20,order=order,milliseconds_42banks=times)
            state['rounds'].append(row);save();print(json.dumps(row),flush=True)
        for entry in immutable:
            t=entry['tensor'];assert t._version==entry['version'] and ahash(t.contiguous().view(torch.uint8).cpu().numpy())==entry['hash']
        assert all(bool((raw[:4096]==113).all() and (raw[-4096:]==113).all()) for raw in guarded)
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['libraries'].items())
        state.update(complete=True,phase='complete',CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),
            full_inputs_native_pack_versions_guards_RNG_unchanged=True,
            additional_native_pack_params_bytes=sum(r['bytes']+r['params_bytes'] for r in state['native_packs']),
            medians_ms={name:statistics.median(r['milliseconds_42banks'][name] for r in state['rounds']) for name in VARIANTS})
        save()
    except BaseException as exc:
        state.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
    finally:
        (GPU/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True));signal.alarm(0)
if __name__=='__main__':main()
