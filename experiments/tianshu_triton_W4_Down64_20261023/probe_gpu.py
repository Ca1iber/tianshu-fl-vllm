# SPDX-License-Identifier: Apache-2.0
import os,signal,subprocess,time,inspect,runpy,importlib,traceback
from common import *

def main():
    signal.alarm(900)
    assert time.time()<1792771080-900 and not GPU.exists()
    GPU.mkdir()
    gate=json.loads((BASE/'CPU_GATE_32g.json').read_text())
    assert gate['complete'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items()) and sha(MODEL)==gate['checkpoint_sha256']
    hardware=subprocess.check_output(['ixsmi'],text=True)
    assert 'No running processes found' in hardware
    (GPU/'hardware_before.txt').write_text(hardware)
    state=dict(complete=False,phase='initializing',pid=os.getpid(),numeric=[],rounds=[],quantization_errors=[],compilation=[],resources={},checkpoint_sha256_before=sha(MODEL),sources=gate['sources'],CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),model_quality_tested=False,official_benchmark=False,adopted=False)
    def save():(GPU/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        profile=json.loads(PROFILE.read_text())
        env={k:v.replace(profile['old_output_root'],str(GPU)) for k,v in profile['environment'].items()}
        os.environ.update(env);sys_path=env['PYTHONPATH'].split(':')
        import sys
        sys.path[:0]=sys_path
        runpy.run_path(str(Path(env['PYTHONPATH'])/'sitecustomize.py'))
        import torch,vllm_fl,flag_gems,kernel
        torch.set_num_threads(4)
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        module=importlib.import_module('_iluvatar.ops.mm')
        state['baseline']=dict(module=module.__name__,source=module.__file__,source_sha256=sha(module.__file__),launch_source=inspect.getsourcefile(module._launch_mm),launch_name=module._launch_mm.__qualname__)
        assert state['baseline']['source_sha256']=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
        assert state['baseline']['launch_source'].endswith('tianshu_prefill_gemm_tiles_20261014/hooks/sitecustomize.py')
        state['environment']=env;save()
        immutable=[];guarded=[]
        def tensor(arr,name):
            arr=np.ascontiguousarray(arr);t=torch.from_numpy(arr.copy())
            if arr.dtype==np.uint16:t=t.view(torch.bfloat16)
            raw=torch.full((arr.nbytes+8192,),113,dtype=torch.uint8,device='cuda')
            view=raw[4096:-4096].view(t.dtype).view(t.shape);view.copy_(t)
            guarded.append(raw);immutable.append((name,view,ahash(arr),int(view._version)))
            return view
        def output_tensor():
            raw=torch.full((64*2048*2+8192,),113,dtype=torch.uint8,device='cuda');guarded.append(raw)
            return raw[4096:-4096].view(torch.bfloat16).view(64,2048)
        banks=[]
        for entry in gate['banks']:
            layer=entry['layer']
            with np.load(entry['file'],allow_pickle=False) as data:
                banks.append(dict(layer=layer,x=tensor(hwords(0,layer),f'X{layer}'),bf16=tensor(np.asarray(weight(layer)),f'BF16{layer}'),w=tensor(data['W'],f'W4{layer}'),s=tensor(data['S'],f'S{layer}'),z=tensor(data['Z'],f'Z{layer}')))
        def bf16_run():return [module.mm(b['x'],b['bf16'].T) for b in banks]
        def w4_run():return [torch.ops.aten._weight_int4pack_mm_dotsaddz.default(b['x'],b['w'],128,b['s'],b['z'],size_n=2048) for b in banks]
        operations={'BF16_qualified':bf16_run,'W4_DOTS_RTN':w4_run}
        for cfg in CONFIGS:
            out=[output_tensor() for _ in banks]
            def custom_run(cfg=cfg,out=out):
                for b,o in zip(banks,out):kernel.run(b['x'],b['w'],b['s'],b['z'],o,cfg)
                return out
            state['phase']='compiling_'+cfg['name'];save()
            try:
                compiled=kernel.run(banks[0]['x'],banks[0]['w'],banks[0]['s'],banks[0]['z'],out[0],cfg)
                torch.cuda.synchronize()
            except BaseException as error:
                if type(error).__name__ in ('OutOfResources','CompilationError','CompileTimeAssertionFailure','UnsupportedLanguageConstruct') and type(error).__module__.startswith('triton'):
                    state['compilation'].append(dict(config=cfg,status='compile_or_resource_failed',error=repr(error),traceback=traceback.format_exc()));save();continue
                raise
            artifacts={};adir=GPU/'compiled'/cfg['name'];adir.mkdir(parents=True)
            for kind,value in compiled.asm.items():
                if isinstance(value,(str,bytes)):
                    path=adir/('kernel.'+kind);path.write_bytes(value.encode() if isinstance(value,str) else value)
                    artifacts[kind]=dict(file=str(path),sha256=sha(path),bytes=path.stat().st_size)
            resource=dict(n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),shared=getattr(getattr(compiled,'metadata',None),'shared',None),artifacts=artifacts)
            state['resources'][cfg['name']]=resource
            state['compilation'].append(dict(config=cfg,status='compiled'));operations[cfg['name']]=custom_run;save()
        state['active_variants']=list(operations);outputs={};graphs={};state['phase']='warming_and_capture';save()
        for name,fn in operations.items():
            for _ in range(2):fn()
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=fn()
            graphs[name]=graph;outputs[name]=out
        state['phase']='numeric';save()
        def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
        def set_phase(phase):
            for b in banks:
                arr=hwords(phase,b['layer']);b['x'].copy_(torch.from_numpy(arr).view(torch.bfloat16))
                for i,e in enumerate(immutable):
                    if e[0]==f"X{b['layer']}":immutable[i]=(e[0],e[1],ahash(arr),int(b['x']._version));break
        for phase in PHASES:
            set_phase(phase);phase_outputs={}
            for name,fn in operations.items():
                eager=fn();torch.cuda.synchronize();expected=[words(t) for t in eager]
                for t in outputs[name]:t.fill_(float('nan'))
                graphs[name].replay();torch.cuda.synchronize();actual=[words(t) for t in outputs[name]]
                assert all(np.array_equal(a,b) for a,b in zip(actual,expected)),(name,phase,'graph_eager')
                assert all(np.isfinite(f32(a)).all() for a in actual),(name,phase,'finite')
                path=GPU/f'{name}_phase{phase}.npz';np.savez(path,**{f'L{layer}':a for layer,a in enumerate(actual)})
                phase_outputs[name]=actual
                state['numeric'].append(dict(phase=phase,variant=name,file=str(path),sha256=sha(path),all_42_graph_poison_eager_exact=True,all_outputs_finite=True));save()
            for layer in range(42):
                original=f32(phase_outputs['BF16_qualified'][layer]).astype(np.float64);quantized=f32(phase_outputs['W4_DOTS_RTN'][layer]).astype(np.float64)
                state['quantization_errors'].append(dict(phase=phase,layer=layer,relative_L2=float(np.linalg.norm(quantized-original)/max(np.linalg.norm(original),1e-30)),max_abs=float(abs(quantized-original).max())))
        set_phase(0);state['phase']='timing';save()
        for mode in ('graph','eager'):
            for rid in range(7):
                order=list(operations);shift=rid%len(order);order=order[shift:]+order[:shift]
                if rid%2:order.reverse()
                times={}
                for name in order:
                    fn=graphs[name].replay if mode=='graph' else operations[name]
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
                    for _ in range(10):fn()
                    end.record();end.synchronize();times[name]=start.elapsed_time(end)/10
                record=dict(mode=mode,round=rid,excluded=rid==0,replays=10,banks_per_replay=42,order=order,milliseconds_42banks=times)
                state['rounds'].append(record);save();print(json.dumps(record),flush=True)
        state['phase']='checking_immutability';save()
        for name,t,expected,version in immutable:
            assert t._version==version and ahash(t.contiguous().view(torch.uint8).cpu().numpy())==expected,name
        for raw in guarded:assert bool((raw[:4096]==113).all() and (raw[-4096:]==113).all())
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        state['RNG_unchanged']=True;state['all_input_weight_scale_zero_full_bytes_versions_guards_preserved']=True
        state['checkpoint_sha256_after']=sha(MODEL)
        assert state['checkpoint_sha256_before']==state['checkpoint_sha256_after'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items())
        state.update(complete=True,phase='complete',checkpoint_unchanged=True,CPU_references_not_yet_verified=True);save()
    except BaseException as error:
        state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
    finally:(GPU/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
