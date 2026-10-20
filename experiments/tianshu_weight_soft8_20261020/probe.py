# SPDX-License-Identifier: Apache-2.0
"""Software W8 component: kernel correctness references decoded weights, not original model quality."""
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import codec
BASE=Path(__file__).resolve().parent
SHAPES=(('qkv',2560,2048),('o',2048,2048),('gateup',12288,2048),('down',2048,6144))
PLANS=[dict(bm=bm,bn=bn,bk=bk,splits=splits,warps=warps,stages=stages)
       for bm,bn,bk,splits,warps,stages in ((64,64,64,1,4,2),(64,128,32,1,4,2),
          (64,128,64,1,4,2),(64,256,32,1,8,2),(64,256,64,1,8,1),
          (64,512,32,1,8,1),(64,128,32,4,4,2),(32,128,64,1,4,2))]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stop-at-utc',required=True)
    args=parser.parse_args()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds())
    assert remaining>1200
    signal.alarm(min(1800,remaining))
    hardware=subprocess.check_output(['ixsmi'],text=True)
    assert 'No running processes found' in hardware,'GPU must be exclusively idle before this probe'
    root=args.output.resolve()
    assert root.is_relative_to(Path('/workspace/benchmark-results')) and not root.exists()
    root.mkdir(parents=True)
    (root/'hardware_before.txt').write_text(hardware)
    import torch,vllm_fl,flag_gems
    from safetensors import safe_open
    import kernel
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    assert getattr(flag_gems,'current_work_registrar',None) is None
    gate=json.loads((BASE/'CPU_GATE_32g.json').read_text())
    assert gate['complete'] and gate['encoding_matches_independent_real_number_rounding']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in gate['sources'].items())
    sources=[Path(__file__),BASE/'codec.py',BASE/'kernel.py',BASE/'CPU_GATE_32g.json',Path(inspect.getsourcefile(flag_gems.mm)),Path('/workspace/MiniCPM5-2B/config.json')]
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in sources:
        target=root/'source_snapshot'/p.relative_to('/')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    result=dict(complete=False,phase='codecs',runner_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
                frozen_sources=frozen,cases=[],weights={},official_benchmark=False,adopted=False,model_executed=False,
                service_executed=False,checkpoint_files_never_modified=True,weight_quantization_changes_numerics=True,
                numerical_reference='FP32 and CPU references use the decoded weights for W8; original BF16 error is separately reported.',
                timing_scope='Actual layer0 weights and synthetic M64 activations, hot Graph GEMM including split reduction, no packing/allocation.')
    def save():
        temporary=root/'result.tmp'
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(root/'result.json')
    def check(out,reference,rows,columns,cpu):
        delta=(out.float()-reference).abs()
        bad=delta>.02+.01*reference.abs()
        actual=out[rows[:,None],columns[None,:]].cpu().float()
        cpu_delta=(actual-cpu).abs()
        cpu_bad=cpu_delta>.02+.01*cpu.abs()
        return dict(passed=bool(torch.isfinite(out).all() and not bad.any() and not cpu_bad.any()),
                    max_abs=float(delta.max()),failed_elements=int(bad.sum()),CPU_max_abs=float(cpu_delta.max()),
                    CPU_failed_elements=int(cpu_bad.sum()),CPU_expected=cpu.tolist(),CPU_actual=actual.tolist(),atol=.02,rtol=.01)
    save()
    try:
        codes_cpu=torch.arange(256,dtype=torch.uint8).repeat(len(codec.BASES))
        bases_cpu=torch.repeat_interleave(torch.tensor(codec.BASES,dtype=torch.int16),256)
        expected_bits=[codec.decode_bits(int(code),int(base)) for code,base in zip(codes_cpu,bases_cpu,strict=True)]
        expected=torch.tensor([bits if bits<32768 else bits-65536 for bits in expected_bits],dtype=torch.int16).view(torch.bfloat16)
        codes,bases=codes_cpu.cuda(),bases_cpu.cuda()
        decoded=torch.empty(codes.shape,device='cuda',dtype=torch.bfloat16)
        kernel.decode_probe[((len(codes)+255)//256,)](codes,bases,decoded,len(codes),256,num_warps=4)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel.decode_probe[((len(codes)+255)//256,)](codes,bases,decoded,len(codes),256,num_warps=4)
        phases=[]
        for phase in (0,1):
            if phase:
                codes.copy_(codes.flip(0))
                bases.copy_(bases.flip(0))
            rng=torch.cuda.get_rng_state().clone()
            graph.replay()
            torch.cuda.synchronize()
            wanted=expected if not phase else expected.flip(0)
            assert torch.equal(decoded.view(torch.int16).cpu(),wanted.view(torch.int16))
            assert torch.equal(torch.cuda.get_rng_state(),rng)
            phases.append(dict(phase=phase,exact_decoded_codes=len(codes),RNG_unchanged=True))
        result['GPU_codecs']=dict(complete=True,changed_data_same_Graph=True,phases=phases)
        del graph,codes,bases,decoded
        files=list(Path('/workspace/MiniCPM5-2B').glob('*.safetensors'))
        assert len(files)==1
        prefix='model.layers.0.'
        keys={'qkv':[prefix+'self_attn.'+s+'_proj.weight' for s in ('q','k','v')],
              'o':[prefix+'self_attn.o_proj.weight'],
              'gateup':[prefix+'mlp.'+s+'_proj.weight' for s in ('gate','up')],
              'down':[prefix+'mlp.down_proj.weight']}
        with safe_open(str(files[0]),framework='pt',device='cpu') as checkpoint:
            weights={name:torch.cat([checkpoint.get_tensor(k) for k in parts],0).contiguous() for name,parts in keys.items()}
        result['phase']='GEMMs'
        for shape_id,(name,n,k) in enumerate(SHAPES):
            assert weights[name].shape==(n,k)
            weight_cpu=weights[name]
            packed_cpu,origins_cpu,stats=codec.prepare_cpu(weight_cpu)
            decoded_cpu=codec.decode_cpu(packed_cpu,origins_cpu)
            stats.update(keys=keys[name],shape=[n,k],BF16_bits_sha256=hashlib.sha256(weight_cpu.view(torch.int16).numpy().tobytes()).hexdigest(),
                         BF16_bytes=weight_cpu.numel()*2,W8_bytes=packed_cpu.numel()+origins_cpu.numel()*2)
            result['weights'][name]=stats
            for index,config in enumerate(PLANS):
                assert n%config['bn']==k%(config['bk']*config['splits'])==0
                torch.manual_seed(202610200900+shape_id)
                x=torch.randn(64,k,device='cuda',dtype=torch.bfloat16)
                w,w8,origin,quant_ref=weight_cpu.cuda(),packed_cpu.cuda(),origins_cpu.cuda(),decoded_cpu.cuda()
                storage={mode:torch.full((67,n),17.,device='cuda',dtype=torch.bfloat16) for mode in ('owned_BF16','owned_W8')}
                partials={mode:torch.full((config['splits']+1,64,n),17.,device='cuda',dtype=torch.float32) for mode in storage} if config['splits']>1 else {}
                scratch={mode:(partials[mode][:config['splits']] if mode in partials else buf[:64],buf[:64]) for mode,buf in storage.items()}
                functions={'native_BF16':lambda:flag_gems.mm(x,w.T),
                           'owned_BF16':lambda:kernel.mm(x,w,origin,scratch['owned_BF16'],config),
                           'owned_W8':lambda:kernel.mm(x,w8,origin,scratch['owned_W8'],config)}
                graphs,outputs,metadata={},{},{}
                row=dict(projection=name,configuration=config,index=index,phases=[])
                try:
                    for mode,function in functions.items():
                        for _ in range(3):
                            function()
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            value=function()
                        graphs[mode]=graph
                        outputs[mode]=value
                        metadata[mode]=dict(kernel.LAST) if mode!='native_BF16' else None
                except Exception as error:
                    row['compile_error']=repr(error)
                    if any(s in str(error).lower() for s in ('illegal memory','device-side assert')):
                        raise
                if len(graphs)==3:
                    rows=torch.linspace(0,63,8).long().cuda()
                    columns=torch.linspace(0,n-1,32).long().cuda()
                    for phase in (0,1,2):
                        if phase==1:
                            x.mul_(.5)
                            w.neg_()
                            w8.bitwise_xor_(128)
                            quant_ref.neg_()
                        elif phase==2:
                            w.mul_(4)
                            origin.add_(16)
                            quant_ref.mul_(4)
                        reference_original=x.float()@w.float().T
                        reference_quant=x.float()@quant_ref.float().T
                        cpu_x=x[rows].cpu().float()
                        cpu_original=cpu_x@w[columns].cpu().float().T
                        cpu_quant=cpu_x@quant_ref[columns].cpu().float().T
                        rng=torch.cuda.get_rng_state().clone()
                        for graph in graphs.values():
                            graph.replay()
                        torch.cuda.synchronize()
                        assert torch.equal(torch.cuda.get_rng_state(),rng)
                        checks={mode:check(out,reference_quant if mode=='owned_W8' else reference_original,rows,columns,cpu_quant if mode=='owned_W8' else cpu_original) for mode,out in outputs.items()}
                        assert all(item['passed'] for item in checks.values()),checks
                        assert all(bool((buf[64:]==17.).all()) for buf in storage.values())
                        assert all(bool((buf[config['splits']:]==17.).all()) for buf in partials.values())
                        approx=outputs['owned_W8'].float()-outputs['native_BF16'].float()
                        row['phases'].append(dict(phase=phase,checks=checks,RNG_unchanged=True,output_partial_guards=True,
                                                  original_BF16_output_relative_L2=float(approx.norm()/outputs['native_BF16'].float().norm()),
                                                  original_BF16_output_max_abs=float(approx.abs().max()),model_quality_not_established=True))
                        if phase==0:
                            times={mode:[] for mode in graphs}
                            for trial in range(7):
                                order=list(graphs) if trial%2==0 else list(reversed(graphs))
                                for mode in order:
                                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                    start.record()
                                    for _ in range(20):
                                        graphs[mode].replay()
                                    end.record()
                                    end.synchronize()
                                    times[mode].append(start.elapsed_time(end)/20)
                            row.update(raw_ms=times,median_ms={mode:statistics.median(values) for mode,values in times.items()})
                    row['compiled_metadata']=metadata
                    row['W8_speedup_percent']=100*(row['median_ms']['native_BF16']/row['median_ms']['owned_W8']-1)
                result['cases'].append(row)
                save()
                print(json.dumps({key:row.get(key) for key in ('projection','index','configuration','median_ms','W8_speedup_percent','compile_error')}),flush=True)
                graphs.clear()
                outputs.clear()
                functions.clear()
                scratch.clear()
                storage.clear()
                partials.clear()
                del x,w,w8,origin,quant_ref
                torch.cuda.empty_cache()
        assert len(result['cases'])==32
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in frozen.items())
        result.update(complete=True,phase='complete',ended_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        import traceback
        result.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        save()

if __name__=='__main__':
    main()
