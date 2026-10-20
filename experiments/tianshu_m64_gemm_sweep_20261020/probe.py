# SPDX-License-Identifier: Apache-2.0
"""M64 BF16 NT targeted scan on real checkpoint layer0; no engine mutation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
BASE = Path(__file__).resolve().parent
KERNEL = BASE.parent/'tianshu_gemm_dot_dtype_20261019/kernel.py'
KERNEL_SHA = 'frozen-at-launch'
SHAPES = (('qkv',2560,2048),('o',2048,2048),('gateup',12288,2048),('down',2048,6144))

def configurations():
    plans = [dict(bm=64,bn=bn,bk=bk,splits=splits,warps=warps,stages=2,group_m=1)
             for bn in (64,128,256) for bk in (32,64) for splits in (1,2,4) for warps in (4,8)]
    plans += [dict(bm=64,bn=512,bk=32,splits=splits,warps=warps,stages=1,group_m=1)
              for splits in (1,2,4) for warps in (8,16)]
    plans += [dict(bm=32,bn=bn,bk=32,splits=splits,warps=4,stages=2,group_m=1)
              for bn in (128,256) for splits in (1,2,4)]
    plans += [dict(bm=64,bn=128,bk=32,splits=8,warps=4,stages=2,group_m=1)]
    assert len(plans) == 49
    assert len({tuple(c.items()) for c in plans}) == 49
    for name,n,k in SHAPES:
        for c in plans:
            assert n % c['bn'] == k % (c['bk']*c['splits']) == 0
    return plans

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stop-at-utc',required=True)
    args=parser.parse_args()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds())
    assert remaining>1200
    signal.alarm(min(remaining,1800))
    root=args.output.resolve()
    assert root.is_relative_to(Path('/workspace/benchmark-results')) and not root.exists()
    root.mkdir(parents=True)
    hardware=subprocess.check_output(['ixsmi'],text=True)
    assert 'No running processes found' in hardware
    (root/'hardware_before.txt').write_text(hardware)
    import torch,vllm_fl,flag_gems
    from safetensors import safe_open
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    assert getattr(flag_gems,'current_work_registrar',None) is None
    spec=importlib.util.spec_from_file_location('_m64_owned_bf16_kernel',KERNEL)
    kernel=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=kernel
    spec.loader.exec_module(kernel)
    vendor=Path(inspect.getsourcefile(flag_gems.mm))
    assert hashlib.sha256(vendor.read_bytes()).hexdigest()=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
    sources=[Path(__file__),KERNEL,vendor,Path('/workspace/MiniCPM5-2B/config.json')]
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in sources:
        target=root/'source_snapshot'/p.relative_to('/')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    plans=configurations()
    result=dict(complete=False,phase='preparing',runner_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
                frozen_sources=frozen,configurations=plans,weights={},cases=[],official_benchmark=False,adopted=False,
                real_checkpoint_weights=True,synthetic_activations=True,BF16_input_weight_output=True,FP32_accumulation=True,
                checkpoint_files_never_modified=True,model_executed=False,service_executed=False,
                scope='Hot single projection graphs, two changed-input phases; includes split reduction, excludes allocations.',
                hardware_properties=str(torch.cuda.get_device_properties(0)))
    def save():
        tmp=root/'result.tmp'
        tmp.write_text(json.dumps(result,indent=2)+'\n')
        tmp.replace(root/'result.json')
    def check(out,ref,rows,cols,cpu):
        delta=(out.float()-ref).abs()
        bad=delta>.02+.01*ref.abs()
        actual=out[rows[:,None],cols[None,:]].cpu().float()
        cdelta=(actual-cpu).abs()
        cbad=cdelta>.02+.01*cpu.abs()
        return dict(passed=bool(torch.isfinite(out).all() and torch.isfinite(ref).all() and not bad.any() and not cbad.any()),
                    max_abs=float(delta.max()),failed_elements=int(bad.sum()),CPU_max_abs=float(cdelta.max()),
                    CPU_failed_elements=int(cbad.sum()),CPU_expected=cpu.tolist(),CPU_actual=actual.tolist(),atol=.02,rtol=.01)
    save()
    try:
        files=list(Path('/workspace/MiniCPM5-2B').glob('*.safetensors'))
        assert len(files)==1
        prefix='model.layers.0.'
        parts={'qkv':[prefix+'self_attn.'+s+'_proj.weight' for s in ('q','k','v')],
               'o':[prefix+'self_attn.o_proj.weight'],
               'gateup':[prefix+'mlp.'+s+'_proj.weight' for s in ('gate','up')],
               'down':[prefix+'mlp.down_proj.weight']}
        with safe_open(str(files[0]),framework='pt',device='cpu') as checkpoint:
            weights={name:torch.cat([checkpoint.get_tensor(key) for key in keys],0).contiguous() for name,keys in parts.items()}
        for name,n,k in SHAPES:
            assert weights[name].shape==(n,k) and weights[name].dtype==torch.bfloat16
            result['weights'][name]=dict(keys=parts[name],shape=[n,k],BF16_bits_sha256=hashlib.sha256(weights[name].view(torch.int16).numpy().tobytes()).hexdigest())
        result['phase']='scan'
        save()
        with torch.inference_mode():
            for shape_id,(name,n,k) in enumerate(SHAPES):
                seed=202610200700+shape_id
                for index,c in enumerate(plans):
                    torch.manual_seed(seed)
                    x=torch.randn(64,k,device='cuda',dtype=torch.bfloat16)
                    w=weights[name].cuda()
                    out_storage=torch.full((67,n),17.,device='cuda',dtype=torch.bfloat16)
                    out=out_storage[:64]
                    partial_storage=torch.full((c['splits']+1,64,n),17.,device='cuda',dtype=torch.float32) if c['splits']>1 else None
                    partial=partial_storage[:c['splits']] if partial_storage is not None else out
                    row=dict(projection=name,shape=[64,n,k],configuration=c,seed=seed,index=index,phases=[])
                    graphs,outputs={},{}
                    fs={'native':lambda:flag_gems.mm(x,w.T),'owned':lambda:kernel.mm(x,w,(partial,out),c,False)}
                    try:
                        for mode in ('native','owned'):
                            for _ in range(3):
                                fs[mode]()
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                actual=fs[mode]()
                            graphs[mode]=graph
                            outputs[mode]=actual
                        row['compiled_metadata']=dict(kernel.LAST)
                    except Exception as error:
                        row.update(compiled=False,error=repr(error))
                        if any(text in str(error).lower() for text in ('illegal memory','device-side assert')):
                            raise
                    if len(graphs)==2:
                        row['compiled']=True
                        rows=torch.linspace(0,63,8).long().cuda()
                        cols=torch.linspace(0,n-1,32).long().cuda()
                        for phase in (0,1):
                            if phase:
                                x.mul_(-.5)
                                w.mul_(2)
                            ref=x.float()@w.float().T
                            cpu=x[rows].cpu().float()@w[cols].cpu().float().T
                            rng_before=torch.cuda.get_rng_state().clone()
                            for graph in graphs.values():
                                graph.replay()
                            torch.cuda.synchronize()
                            assert torch.equal(torch.cuda.get_rng_state(),rng_before)
                            checks={mode:check(value,ref,rows,cols,cpu) for mode,value in outputs.items()}
                            assert all(v['passed'] for v in checks.values()),checks
                            assert bool((out_storage[64:]==17.).all())
                            if partial_storage is not None:
                                assert bool((partial_storage[c['splits']:]==17.).all())
                            row['phases'].append(dict(phase=phase,checks=checks,RNG_unchanged=True,output_and_partial_guards=True,
                                                      exact_to_native=bool(torch.equal(outputs['owned'],outputs['native']))))
                        times={'native':[],'owned':[]}
                        for trial in range(7):
                            for mode in (('native','owned') if trial%2==0 else ('owned','native')):
                                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                begin.record()
                                for _ in range(20):
                                    graphs[mode].replay()
                                end.record()
                                end.synchronize()
                                times[mode].append(begin.elapsed_time(end)/20)
                        row.update(raw_ms=times,median_ms={mode:statistics.median(values) for mode,values in times.items()})
                        row['speedup_percent']=100*(row['median_ms']['native']/row['median_ms']['owned']-1)
                    result['cases'].append(row)
                    save()
                    print(json.dumps({key:row.get(key) for key in ('projection','index','compiled','configuration','median_ms','speedup_percent','error')}),flush=True)
                    graphs.clear()
                    outputs.clear()
                    fs.clear()
                    del x,w,out,partial,out_storage,partial_storage,actual
                    torch.cuda.empty_cache()
        assert len(result['cases'])==196
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in frozen.items())
        best={name:min((row for row in result['cases'] if row['projection']==name and row.get('compiled')),key=lambda row:row['median_ms']['owned']) for name,n,k in SHAPES}
        result.update(complete=True,phase='complete',best_screen_candidates={name:dict(configuration=row['configuration'],median_ms=row['median_ms'],speedup_percent=row['speedup_percent']) for name,row in best.items()},
                      ended_utc=datetime.now(timezone.utc).isoformat(),selection_bias='Per-projection minima on discovery data; requires held-out validation and actual service measurement.')
    except BaseException as error:
        import traceback
        result.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        save()
    return 0 if result['complete'] else 1

if __name__=='__main__':
    raise SystemExit(main())
