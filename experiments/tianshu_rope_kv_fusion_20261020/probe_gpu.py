# SPDX-License-Identifier: Apache-2.0
"""Independent component: compiled native RoPE+writer versus fused variants.

No model patch. Explicit native BF16-product reference validates fused arithmetic;
compiler precision differences are recorded, not called model accuracy failures.
"""
import argparse,datetime,hashlib,json,os,signal,statistics,sys,time,traceback
from pathlib import Path

BASE=Path(__file__).resolve().parent
MODES=('compiled_native','fused_w1','fused_w2','fused_w4')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,default=str)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stop-at-utc',required=True);args=parser.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    deadline=datetime.datetime.fromisoformat(args.stop_at_utc).timestamp()
    remaining=int(deadline-time.time());assert remaining>120
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('GPU reservation deadline')))
    signal.alarm(remaining)
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(root/'inductor_cache')
    os.environ['TRITON_CACHE_DIR']=str(root/'triton_cache')
    state=dict(complete=False,phase='imports',pid=os.getpid(),cases=[],service_or_Math_run=False,
        modes=list(MODES),not_adopted=True,stop_at_utc=args.stop_at_utc,
        native_baseline='torch.compile(fullgraph=True,dynamic=False,backend=inductor) actual RotaryEmbedding.forward_static + qualified writer')
    write(root/'state.json',state)
    try:
        import importlib.util,inspect
        import numpy as np
        import torch
        import flag_gems
        import kernel
        from check_cpu import encode,scalar_rotation
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
        sys.path.insert(0,str(BASE.parent/'tianshu_k8v16_pipeline_20261019'))
        from value_planes import value_plane,clear_cache
        spec=importlib.util.spec_from_file_location('_rope_fusion_current_writer',BASE.parent/'tianshu_soft8_mixed_planes_20261019/writer.py')
        writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer)
        torch.set_num_threads(1)
        gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['CPU_only'] and gate['count']==14
        for path,sha in gate['sources'].items():assert digest(path)==sha
        sources=dict(gate['sources'])
        for line in (BASE/'SOURCE_PREREQUISITE.sha256').read_text().splitlines():
            sha,path=line.split(maxsplit=1);path=str(Path(path).resolve());assert digest(path)==sha;sources[path]=sha
        for path in (Path(__file__),BASE/'CPU_GATE_32g.json',BASE/'SOURCE_PREREQUISITE.sha256',BASE/'verify_gpu.py',BASE/'run_after_official.sh'):
            sources[str(path)]=digest(path)
        for path in sources:
            dest=root/'source_snapshot'/Path(path).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(Path(path).read_bytes());assert digest(dest)==sources[path]
        state.update(sources=sources,torch_version=torch.__version__,GPU=torch.cuda.get_device_name(),
            actual_reference_source=inspect.getsourcefile(RotaryEmbedding.forward_static),CPU_cases=14)
        flag_gems.enable(record=False,once=True)
        torch.cuda.synchronize();rng=torch.cuda.get_rng_state().clone()
        gen=torch.Generator().manual_seed(202610204312)
        cc=torch.randn((131072,128),generator=gen,dtype=torch.float32).bfloat16()
        cache=cc.cuda();np.save(root/'cos_sin_bf16_bits.npy',cc.view(torch.uint16).numpy())
        state['cache_artifact_sha256']=digest(root/'cos_sin_bf16_bits.npy')
        native=lambda p,q,k,c:RotaryEmbedding.forward_static(p,q,k,128,128,c,True)
        for n in (1,8,32,40,41,63,64):
          for stride in (2560,2688):
            clear_cache();torch._dynamo.reset()
            name=f'B{n}_stride{stride}';folder=root/name;folder.mkdir()
            blocks=(n*17+31)//16+2;byte_count=blocks*12288+256
            inputs=torch.empty((n,stride),device='cuda',dtype=torch.bfloat16)
            q,k,v=inputs[:,:2048],inputs[:,2048:2304],inputs[:,2304:2560]
            positions=torch.zeros(n,device='cuda',dtype=torch.int64);slots=positions.clone()
            compiled=torch.compile(native,backend='inductor',fullgraph=True,dynamic=False)
            holders={};graphs={};raws={};qguards={};kguards={};metadata={}
            for mode in MODES:
                raw=torch.full((byte_count,),173,device='cuda',dtype=torch.uint8);raws[mode]=raw
                kc=torch.as_strided(raw,(blocks,16,2,128),(12288,256,128,1),storage_offset=128)
                rv=torch.as_strided(raw,(blocks,16,2,128),(12288,256,128,1),storage_offset=128+4096)
                vc=value_plane(rv);holders[mode]=dict(kc=kc,vc=vc)
                if mode!='compiled_native':
                    qguards[mode]=torch.full((n*2048+256,),9.25,device='cuda',dtype=torch.bfloat16)
                    kguards[mode]=torch.full((n*256+256,),9.25,device='cuda',dtype=torch.bfloat16)
                    holders[mode].update(q=qguards[mode][128:-128].view(n,2048),k=kguards[mode][128:-128].view(n,256))
            first=torch.randn((n,stride),generator=gen,dtype=torch.float32).bfloat16()
            inputs.copy_(first);positions.fill_(0);slots.copy_(torch.arange(n,dtype=torch.int64,device='cuda'))
            def execute(mode):
                holder=holders[mode]
                if mode=='compiled_native':
                    oq,ok=compiled(positions,q,k,cache);holder.update(q=oq,k=ok)
                    writer.write(ok.view(n,2,128),v.view(n,2,128),holder['kc'],holder['vc'],slots)
                else:
                    compiled_kernel=kernel.launch(positions,q,k,v,cache,slots,holder['q'],holder['k'],holder['kc'],holder['vc'],int(mode[-1]))
                    metadata[mode]=dict(kernel_hash=getattr(compiled_kernel,'hash',None),n_regs=getattr(compiled_kernel,'n_regs',None),
                        n_spills=getattr(compiled_kernel,'n_spills',None),num_warps=compiled_kernel.metadata.num_warps,
                        num_stages=compiled_kernel.metadata.num_stages,shared=compiled_kernel.metadata.shared,
                        grid=[n,18],fp_fusion_disabled=True,constants={str(key):str(value) for key,value in compiled_kernel.src.constants.items()})
            for mode in MODES:
                execute(mode);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(16):execute(mode)
                graphs[mode]=graph
            case=dict(name=name,batch=n,input_row_stride=stride,phases=[],kernel_metadata=metadata)
            for phase in range(3):
                cpu=torch.randn((n,stride),generator=gen,dtype=torch.float32).bfloat16()
                pos=torch.randint(0,131072,(n,),generator=gen,dtype=torch.int64);pos[-1]=131071-phase
                if phase==2:pos=4096+torch.arange(n,dtype=torch.int64)
                slot=torch.arange(n,dtype=torch.int64)*17+15-phase
                if phase<2:slot[(torch.arange(n)+phase)%5==0]=-1
                inputs.copy_(cpu);positions.copy_(pos);slots.copy_(slot)
                for raw in raws.values():raw.fill_(173)
                expected_q=scalar_rotation(cpu[:,:2048],pos,cc)
                expected_k=scalar_rotation(cpu[:,2048:2304],pos,cc)
                expected_eager_q,expected_eager_k=RotaryEmbedding.forward_static(pos,cpu[:,:2048],cpu[:,2048:2304],128,128,cc,True)
                assert torch.equal(expected_q,expected_eager_q) and torch.equal(expected_k,expected_eager_k)
                artifacts=dict(input_bf16=cpu.view(torch.uint16).numpy(),positions=pos.numpy(),slots=slot.numpy(),
                    expected_q_bf16=expected_q.view(torch.uint16).numpy(),expected_k_bf16=expected_k.view(torch.uint16).numpy())
                observed={}
                for mode in MODES:
                    graphs[mode].replay();torch.cuda.synchronize()
                    oq=holders[mode]['q'].cpu();ok=holders[mode]['k'].cpu();actual=raws[mode].cpu().numpy()
                    assert torch.isfinite(oq).all() and torch.isfinite(ok).all()
                    if mode!='compiled_native':
                        assert torch.equal(oq,expected_q) and torch.equal(ok,expected_k),(name,phase,mode,'CPU rounded mismatch')
                        assert torch.all(qguards[mode][:128]==9.25) and torch.all(qguards[mode][-128:]==9.25)
                        assert torch.all(kguards[mode][:128]==9.25) and torch.all(kguards[mode][-128:]==9.25)
                    expected=np.full(byte_count,173,dtype=np.uint8)
                    ek=encode(ok.reshape(n,2,128)).numpy();ev=cpu[:,2304:2560].contiguous().view(torch.uint8).numpy().reshape(n,512)
                    for r,value in enumerate(slot.tolist()):
                        if value<0:continue
                        b,t=divmod(value,16);off=128+b*12288+t*256;expected[off:off+256]=ek[r].reshape(-1)
                        off=128+b*12288+4096+t*512;expected[off:off+512]=ev[r]
                    assert np.array_equal(actual,expected),(name,phase,mode,'full physical cache mismatch')
                    assert torch.equal(inputs.cpu(),cpu) and torch.equal(positions.cpu(),pos) and torch.equal(slots.cpu(),slot)
                    artifacts[mode+'_q']=oq.view(torch.uint16).numpy();artifacts[mode+'_k']=ok.view(torch.uint16).numpy();artifacts[mode+'_raw']=actual
                    observed[mode]=dict(q_maxabs_vs_native_eager=float((oq.float()-expected_q.float()).abs().max()),
                        k_maxabs_vs_native_eager=float((ok.float()-expected_k.float()).abs().max()),
                        bitwise_vs_native_eager=bool(torch.equal(oq,expected_q) and torch.equal(ok,expected_k)),
                        full_cache_and_guards_match=True,inputs_unchanged=True)
                artifact=folder/f'phase{phase}.npz';np.savez_compressed(artifact,**artifacts)
                case['phases'].append(dict(phase=phase,artifact=str(artifact),sha256=digest(artifact),modes=observed))
            timing={mode:[] for mode in MODES}
            for repeat in range(7):
                order=MODES if repeat%2==0 else tuple(reversed(MODES))
                for mode in order:
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100):graphs[mode].replay()
                    end.record();end.synchronize();timing[mode].append(start.elapsed_time(end)/100/16)
            case['timing_ms']=timing;case['median_ms']={mode:statistics.median(samples) for mode,samples in timing.items()}
            case['compiled_baseline_vs_fused_bitwise_all_phases']=all(row['modes']['compiled_native']['bitwise_vs_native_eager'] for row in case['phases'])
            state['cases'].append(case);state['phase']='component_running';write(root/'state.json',state)
            print(json.dumps(dict(case=name,median_ms=case['median_ms'],compiled_baseline_exact=case['compiled_baseline_vs_fused_bitwise_all_phases'])),flush=True)
            del graphs,holders,raws,qguards,kguards,compiled,inputs,q,k,v
            clear_cache();torch.cuda.empty_cache()
        assert torch.equal(cache.cpu(),cc) and torch.equal(torch.cuda.get_rng_state(),rng)
        assert all(digest(path)==sha for path,sha in sources.items())
        compiled_sources={str(p):digest(p) for p in (root/'inductor_cache').rglob('*.py')}
        assert compiled_sources,'No emitted inductor Python sources found'
        state.update(complete=True,phase='complete',case_count=14,phase_count=42,mode_checks=168,
            compiled_inductor_sources=compiled_sources,RNG_unchanged=True,cos_sin_cache_unchanged=True,
            timing_groups=392,iterations_per_timing_group=100,components_per_graph=16,full_model_quality_not_tested=True,timing_uses_all_valid_slots=True,timing_positions_start_at_4096=True)
        write(root/'result.json',state);write(root/'state.json',state)
        (root/'complete.txt').write_text('complete\n')
    except BaseException as error:
        state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());write(root/'state.json',state);raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
