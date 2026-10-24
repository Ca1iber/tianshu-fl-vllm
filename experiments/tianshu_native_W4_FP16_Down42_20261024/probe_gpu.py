# SPDX-License-Identifier: Apache-2.0
"""42 distinct current W4 banks, native FP16 raw and actual casting pipeline."""
import os,signal,subprocess,statistics,traceback
from datetime import datetime,timezone
from common import *
def main():
    remaining=int((datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds())
    assert remaining>1200;signal.alarm(min(1800,remaining));GPU.mkdir(exist_ok=False)
    state=dict(complete=False,phase='initializing',pid=os.getpid(),numeric=[],rounds=[],
        performance_scope='42realDown banks; casting measured in full variant; rawFP16 diagnostic only',
        model_executed=False,official_benchmark=False,adopted=False)
    def save():
        p=GPU/'result.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(GPU/'result.json')
    save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
        (GPU/'hardware_before.txt').write_text(hardware)
        gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['current_r5_Down_all42_payloads_exact']
        assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['libraries'].items())
        assert all(sha(p)==h for p,h in gate['files'].items())
        import torch,vllm_fl
        torch.set_num_threads(4)
        guarded=[];immutable=[]
        def tensor(array,dtype,name):
            array=np.ascontiguousarray(array);cpu=torch.from_numpy(array.copy())
            if array.dtype==np.uint16:cpu=cpu.view(dtype)
            raw=torch.full((array.nbytes+8192,),113,dtype=torch.uint8,device='cuda')
            t=raw[4096:-4096].view(cpu.dtype).reshape(cpu.shape);t.copy_(cpu)
            guarded.append(raw);immutable.append(dict(name=name,tensor=t,hash=ahash(array),version=int(t._version)))
            return t
        banks=[]
        for bank in gate['banks']:
            layer=bank['layer'];a=np.load(bank['file'],allow_pickle=False);h=parent.hwords(0,layer);xh=f32(h).astype(np.float16)
            banks.append(dict(layer=layer,x=tensor(h,torch.bfloat16,f'X{layer}'),xh=tensor(xh.view(np.uint16),torch.float16,f'XH{layer}'),
                w=tensor(a['W'],None,f'W{layer}'),s=tensor(a['S'],torch.bfloat16,f'S{layer}'),z=tensor(a['Z'],torch.bfloat16,f'Z{layer}'),
                sh=tensor(f32(a['S']).astype(np.float16).view(np.uint16),torch.float16,f'SH{layer}'),
                zh=tensor(f32(a['Z']).astype(np.float16).view(np.uint16),torch.float16,f'ZH{layer}')))
        def call(x,b,half):
            return torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,b['w'],128,b['sh'] if half else b['s'],b['zh'] if half else b['z'],size_n=2048)
        operations={
            VARIANTS[0]:lambda:[call(b['x'],b,False) for b in banks],
            VARIANTS[1]:lambda:[call(b['xh'],b,True) for b in banks],
            VARIANTS[2]:lambda:[call(b['x'].to(torch.float16),b,True).to(torch.bfloat16) for b in banks]}
        graphs={};outputs={}
        state['phase']='warming';save()
        for name,fn in operations.items():
            for _ in range(3):fn()
            torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=fn()
            graphs[name]=g;outputs[name]=out
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        def phase_inputs(phase):
            for b in banks:
                layer=b['layer'];h=parent.hwords(phase,layer);xh=f32(h).astype(np.float16).view(np.uint16)
                for prefix,array,key,dtype in [('X',h,'x',torch.bfloat16),('XH',xh,'xh',torch.float16)]:
                    t=b[key];t.copy_(torch.from_numpy(array).view(dtype))
                    entry=next(v for v in immutable if v['name']==f'{prefix}{layer}')
                    entry.update(hash=ahash(array),version=int(t._version))
        for phase in PHASES:
            phase_inputs(phase);state['phase']=f'numeric{phase}';save()
            for name,fn in operations.items():
                eager=fn();torch.cuda.synchronize()
                expected=[t.contiguous().view(torch.uint16).cpu().numpy().copy() for t in eager]
                repeated=fn();torch.cuda.synchronize()
                repeat_exact=all(np.array_equal(a,t.contiguous().view(torch.uint16).cpu().numpy()) for a,t in zip(expected,repeated))
                for t in outputs[name]:t.fill_(float('nan'))
                graphs[name].replay();torch.cuda.synchronize()
                actual=[t.contiguous().view(torch.uint16).cpu().numpy().copy() for t in outputs[name]]
                path=GPU/f'{name}_phase{phase}.npz';np.savez(path,**{f'L{layer}':array for layer,array in enumerate(actual)})
                checks=[]
                for ref in gate['references']:
                    if ref['phase']!=phase:continue
                    layer=ref['layer'];reference=np.load(ref['file'],allow_pickle=False)
                    raw=actual[layer];kind='FP16' if name==VARIANTS[1] else 'BF16'
                    values=raw.view(np.float16).astype(np.float32) if kind=='FP16' else f32(raw)
                    target=reference['BF16' if name==VARIANTS[0] else 'FP16']
                    check=metrics(values,target,kind);checks.append(dict(layer=layer,**check))
                exact=all(np.array_equal(a,b) for a,b in zip(expected,actual))
                finite=all(np.isfinite(a.view(np.float16) if name==VARIANTS[1] else f32(a)).all() for a in actual)
                row=dict(phase=phase,variant=name,file=str(path),sha256=sha(path),selected_FP64=checks,
                    all42_repeat_eager_exact=repeat_exact,all42_poison_Graph_eager_exact=exact,all42_finite=bool(finite),
                    actual_dtype=str(outputs[name][0].dtype))
                state['numeric'].append(row);save()
                assert repeat_exact and exact and finite and all(c['passed'] for c in checks),row
        phase_inputs(0)
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        state['phase']='timing';save()
        for round_id in range(9):
            order=list(operations);order.reverse() if round_id%2 else None;times={}
            for name in order:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
                for _ in range(20):graphs[name].replay()
                end.record();end.synchronize();times[name]=start.elapsed_time(end)/20
            row=dict(round=round_id,replays=20,order=order,milliseconds_42banks=times)
            state['rounds'].append(row);save();print(json.dumps(row),flush=True)
        for entry in immutable:
            t=entry['tensor'];assert t._version==entry['version'] and ahash(t.contiguous().view(torch.uint8).cpu().numpy())==entry['hash']
        assert all(bool((raw[:4096]==113).all() and (raw[-4096:]==113).all()) for raw in guarded)
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['libraries'].items())
        state.update(complete=True,phase='complete',CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),
            full_inputs_versions_guards_RNG_unchanged=True,
            actual_added_FP16_scale_zero_bytes=sum((b['sh'].numel()+b['zh'].numel())*2 for b in banks),
            medians_ms={name:statistics.median(r['milliseconds_42banks'][name] for r in state['rounds']) for name in VARIANTS})
        save()
    except BaseException as exc:
        state.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
    finally:
        (GPU/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True));signal.alarm(0)
if __name__=='__main__':main()
