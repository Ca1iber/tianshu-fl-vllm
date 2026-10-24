# SPDX-License-Identifier: Apache-2.0
"""Actual current K8V16 versus dynamicV8 floating/integer full writer+attention."""
import argparse,hashlib,importlib.util,inspect,json,os,signal,statistics,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import codec,domain_gpu,launch_sweep,failure_evidence

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-at-utc',required=True)
    args=parser.parse_args()
    remaining=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds())
    assert remaining>1200;signal.alarm(min(1800,remaining))
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results')
    root.mkdir(exist_ok=False)
    state=dict(complete=False,phase='initializing',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        cases=[],codec_validation={},primitive_records=[],declared_grid=launch_sweep.CONFIGS,
        engine_allocator_connected=False,prefill_executed=False,model_executed=False,
        service_executed=False,official_benchmark=False,adopted=False)
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');tmp.replace(root/'result.json')
    save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True)
        assert 'Iluvatar BI-V150' in hardware and 'No running processes found' in hardware
        (root/'hardware_before.txt').write_text(hardware)
        base=Path(__file__).resolve().parent;gate=json.loads((base/'CPU_GATE_32g.json').read_text())
        assert gate['complete'] and gate['CPU_only']
        assert all(sha(path)==h for path,h in gate['sources'].items())
        import torch,triton,vllm_fl
        from vllm.platforms import current_platform
        from vllm.v1.kv_cache_interface import FullAttentionSpec,KVQuantMode
        from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
        assert current_platform.vendor_name=='iluvatar' and torch.cuda.device_count()==1
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        launch_dir=base.parent/'tianshu_k8v16_address32_20261020';sys.path.insert(0,str(launch_dir))
        component=load('_current_PV_component',base/'component_launch.py')
        layout=load('_current_PV_layout',codec.LAYOUT)
        reference_quant=load('_current_PV_reference_quant',codec.REFERENCE)
        writer=load('_current_PV_writer',codec.WRITER)
        helper=codec.load_helper()
        generated=root/'generated';generated.mkdir()
        models=[dict(label='control',reader=load('_PV_current_control',codec.READER),page_bytes=12288)]
        proof={}
        for cfg in launch_sweep.CONFIGS:
            text,proof[cfg['label']]=codec.render_reader(codec.READER.read_bytes(),cfg['variant'])
            path=generated/f"reader_{cfg['label']}.py";path.write_text(text)
            models.append(dict(label=cfg['label'],reader=load('_PV_'+cfg['label'],path),config=cfg,page_bytes=8448))
        for model in models:model['invoke'],model['producer']=launch_sweep.attach(model,component,root/'codegen')
        paths=set(map(Path,gate['sources']))|{base/'CPU_GATE_32g.json',Path(component.__file__),component.LEGACY,component.POLICY,
            launch_dir/'guard.py',launch_dir/'source.py',Path(inspect.getsourcefile(FullAttentionSpec)),
            Path(inspect.getsourcefile(_reshape_attention_kv_cache)),Path(torch.__file__),Path(triton.__file__)}
        paths|={Path(model['reader'].__file__) for model in models}
        frozen={str(p.resolve()):sha(p) for p in paths}
        for path,h in frozen.items():
            target=root/'source_snapshot'/Path(path).relative_to('/')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==h
        state.update(frozen_sources=frozen,source_proofs=proof,launch_proof=component.source_proof(),
            torch_version=torch.__version__,triton_version=triton.__version__,GPU=torch.cuda.get_device_name(0),
            compiled_kernel_records={m['label']:m['producer']._sweep_codegen.records for m in models},
            time_scope='one actual writer incl quantization/scales plus one paged attention;9alternating trials x20Graph')
        save()
        def error(actual,expected):
            actual,expected=(actual.double(),expected.double()) if actual.device.type=='cpu' else (actual.float(),expected.float())
            delta=(actual-expected).abs();failed=delta>.002+.01*expected.abs()
            return dict(passed=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all() and not failed.any()),
                max_abs=delta.max().item(),failed_elements=int(failed.sum()),atol=.002,rtol=.01)
        def quantize_k(x):
            bits=x.view(torch.int16).int();mag=bits&32767
            code=((mag+7+((mag>>4)&1))>>4)-928
            return (torch.where(code>=8,code.clamp(max=127),0)|((bits&32768)>>8)).byte()
        def decode_k(x):
            bits=x.int();mag=bits&127
            return (torch.where(mag>=8,(mag<<4)+14848,0)|((bits&128)<<8)).short().view(torch.bfloat16)
        def reference(q,k,v,used):
            n,width=k.shape[:2]
            query=q.float().reshape(n,2,8,128)
            scores=torch.einsum('bhgd,bkhd->bhgk',query,k.float())*128**-.5
            mask=torch.arange(width,device=q.device)[None,:]<used[:,None]
            scores.masked_fill_(~mask[:,None,None,:],-float('inf'))
            return torch.einsum('bhgk,bkhd->bhgd',scores.softmax(-1),v.float()).reshape(n,16,128)
        def cpu_checks(q,k,v,lengths,actual):
            checks=[]
            for seq in sorted({0,q.shape[0]//2,q.shape[0]-1}):
                key=k[seq,:lengths[seq]].cpu().double();value=v[seq,:lengths[seq]].cpu().double()
                for head in (0,7,8,15):
                    query=q[seq,head].cpu().double()
                    expected=((key[:,head//8]@query)*128**-.5).softmax(0)@value[:,head//8]
                    check=error(actual[seq,head].cpu(),expected);check.update(sequence=seq,head=head,scalars=128)
                    assert check['passed'],check
                    checks.append(check)
            return checks
        with torch.inference_mode():
            state['phase']='primitive grid validation';save()
            state['codec_validation'],state['primitive_records']=domain_gpu.validate(helper,root,torch);save()
            cases=[('decode64_4096',[4096]*64,False),('decode64_5120',[5120]*64,False),
                   ('decode64_ragged_strided',[4096-(i%7)*131 for i in range(64)],True)]
            for case_id,(name,lengths,strided) in enumerate(cases):
                state['phase']=name;save()
                seed=202610241100+case_id;torch.manual_seed(seed)
                n=len(lengths);width=(max(lengths)+15)//16;pages=n*width
                logical_k=torch.randn(n,width*16,2,128,device='cuda',dtype=torch.bfloat16)
                logical_v=torch.randn_like(logical_k);logical_v[0,0,0].zero_()
                qs=torch.randn(n,20 if strided else 16,128,device='cuda',dtype=torch.bfloat16);q=qs[:,:16]
                ts=torch.zeros(n,width+3 if strided else width,device='cuda',dtype=torch.int32);table=ts[:,:width]
                table.copy_(torch.randperm(pages,device='cuda').int().reshape(n,width))
                used=torch.tensor(lengths,device='cuda',dtype=torch.int32);cu=torch.arange(n+1,device='cuda',dtype=torch.int32)
                fixtures=[]
                for model in models:
                    physical=(pages+2)*(2 if strided else 1)
                    mode='K8V16' if model['label']=='control' else 'K8VI8Heads'
                    raw,native,k,v,ks,vs,geometry=layout.allocate(torch,FullAttentionSpec,_reshape_attention_kv_cache,mode,physical,strided)
                    oslab=torch.full((n+13,20 if strided else 16,128),17.,device='cuda',dtype=torch.bfloat16);out=oslab[:n,:16]
                    kw=dict(q=q,k=k,v=v,out=out,cu_seqlens_q=cu,max_seqlen_q=1,seqused_k=used,max_seqlen_k=width*16,
                        softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=table,softcap=0.,
                        q_descale=None,k_descale=None,v_descale=None,seq_threshold_3D=64,num_par_softmax_segments=4,
                        softmax_segm_output=torch.full((64,16,64,128),17.,device='cuda'),
                        softmax_segm_max=torch.full((64,16,64),17.,device='cuda'),
                        softmax_segm_expsum=torch.full((64,16,64),17.,device='cuda'),
                        kv_quant_mode=KVQuantMode.NONE if mode=='K8V16' else KVQuantMode.INT8_PER_TOKEN_HEAD,
                        k_scale_cache=ks,v_scale_cache=vs)
                    fixtures.append(dict(model=model,raw=raw,native=native,k=k,v=v,ks=ks,vs=vs,geometry=geometry,
                        out=out,out_storage=oslab,kw=kw,graph=None,slots=None,key=None,value=None))
                phases=[]
                for phase in range(2):
                    if phase:
                        logical_k.mul_(.5);logical_v.neg_();q.neg_()
                        lengths=[length-min(7,length-1) for length in lengths]
                        used.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
                        table.copy_(torch.randperm(pages,device='cuda').int().reshape(n,width))
                    inputs=[logical_k,logical_v,qs,ts,used];backups=[t.clone() for t in inputs]
                    encoded_k=quantize_k(logical_k);reference_k=decode_k(encoded_k)
                    encoded_v,vscales=reference_quant.quantize(logical_v)
                    reference_v=encoded_v.float()*vscales[...,None]
                    checks=[]
                    for fixture in fixtures:
                        candidate=fixture['model']['label']!='control'
                        stored_v=encoded_v if candidate else logical_v
                        answer_v=reference_v if candidate else logical_v
                        ids=table.flatten().long()
                        fixture['k'][ids]=encoded_k.reshape(pages,16,2,128);fixture['v'][ids]=stored_v.reshape(pages,16,2,128)
                        if candidate:
                            fixture['ks'][ids]=1.;fixture['vs'][ids]=vscales.reshape(pages,16,2)
                        expected_storage=fixture['raw'].clone()
                        pos=used.long()-1;arange=torch.arange(n,device='cuda')
                        slots=table[arange,pos//16].long()*16+pos%16
                        slots=torch.cat((slots,torch.tensor([-1,-33,-999],device='cuda',dtype=torch.int64)))
                        key=torch.cat((logical_k[arange,pos],torch.full((3,2,128),19.,device='cuda',dtype=torch.bfloat16)))
                        value=torch.cat((logical_v[arange,pos],torch.full((3,2,128),23.,device='cuda',dtype=torch.bfloat16)))
                        if fixture['slots'] is None:fixture.update(slots=slots,key=key,value=value)
                        else:
                            fixture['slots'].copy_(slots);fixture['key'].copy_(key);fixture['value'].copy_(value)
                        blocks,tokens=slots[:n]//16,slots[:n]%16
                        fixture['k'][blocks,tokens]=255;fixture['v'][blocks,tokens]=117 if candidate else -13.
                        if candidate:fixture['ks'][blocks,tokens]=111.;fixture['vs'][blocks,tokens]=333.
                        def invoke(f=fixture):
                            writer.write(f['key'],f['value'],f['k'],f['v'],f['slots'],f['ks'],f['vs'])
                            f['model']['invoke'](f['kw'])
                        if not phase:
                            for _ in range(3):invoke()
                            fixture['graph']=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(fixture['graph']):invoke()
                        extra=[fixture['slots'],fixture['key'],fixture['value']];extra_backups=[t.clone() for t in extra]
                        rng=torch.cuda.get_rng_state().clone();crng=torch.get_rng_state().clone()
                        fixture['graph'].replay();torch.cuda.synchronize()
                        assert torch.equal(torch.cuda.get_rng_state(),rng) and torch.equal(torch.get_rng_state(),crng)
                        assert torch.equal(fixture['raw'],expected_storage)
                        assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(inputs,backups))
                        assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(extra,extra_backups))
                        assert bool((fixture['out_storage'][n:]==17.).all())
                        if strided:assert bool((fixture['out_storage'][:n,16:]==17.).all())
                        segments=fixture['model']['producer'].last_record['segments']
                        for field,factor in (('softmax_segm_output',128),('softmax_segm_max',1),('softmax_segm_expsum',1)):
                            assert bool((fixture['kw'][field].flatten()[n*16*segments*factor:]==17.).all())
                        expected=reference(q,reference_k,answer_v,used)
                        numerical=error(fixture['out'],expected)
                        if not numerical['passed']:
                            evidence=failure_evidence.save(root,name,fixture['model']['label'],phase,q,reference_k,answer_v,used,fixture['out'],expected)
                            state.setdefault('numeric_failures',[]).append(evidence);save()
                        assert numerical['passed'],(name,fixture['model']['label'],numerical)
                        cpu=cpu_checks(q,reference_k,answer_v,lengths,fixture['out'])
                        writer_fixture=None
                        if candidate:
                            path=root/f"writer_{name}_{fixture['model']['label']}_phase{phase}.npz"
                            np.savez(path,value_words=fixture['value'][:n].view(torch.uint16).cpu().numpy(),
                                actual_V=fixture['v'][blocks,tokens].cpu().numpy(),actual_scale=fixture['vs'][blocks,tokens].cpu().numpy(),
                                actual_K_scale=fixture['ks'][blocks,tokens].cpu().numpy())
                            writer_fixture=dict(file=str(path),sha256=sha(path))
                        checks.append(dict(label=fixture['model']['label'],numerical=numerical,CPU_FP64=cpu,
                            full_cache_storage_bitwise_exact=True,output_and_scratch_guards=True,RNG_unchanged=True,
                            full_inputs_bitwise_unchanged=True,quantized_V_reference=candidate,
                            physical_page_bytes=fixture['model']['page_bytes'],writer_fixture=writer_fixture,
                            V_quantization_max_abs=(answer_v.float()-logical_v.float()).abs().max().item()))
                    phases.append(dict(phase=phase,lengths=list(lengths),checks=checks,
                        same_Graph_Q_K_V_scales_table_lengths_slots_changed=bool(phase)))
                measurements=[[] for _ in fixtures]
                for trial in range(9):
                    order=range(len(fixtures)) if trial%2==0 else reversed(range(len(fixtures)))
                    for index in order:
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
                        for _ in range(20):fixtures[index]['graph'].replay()
                        end.record();end.synchronize();measurements[index].append(start.elapsed_time(end)/20)
                medians=list(map(statistics.median,measurements))
                record=dict(name=name,seed=seed,strided=strided,batch=n,phases=phases,labels=[m['label'] for m in models],
                    raw_ms=measurements,median_ms=medians,candidate_time_reduction_percent=[100*(1-t/medians[0]) for t in medians[1:]],
                    actual_launch=[m['producer'].last_record for m in models],layouts=[f['geometry'] for f in fixtures])
                state['cases'].append(record);save()
                print(json.dumps({k:record[k] for k in ('name','median_ms','candidate_time_reduction_percent')}),flush=True)
                del fixtures,fixture,expected_storage,expected,logical_k,logical_v,reference_k,reference_v,encoded_k,encoded_v,answer_v,stored_v
                torch.cuda.empty_cache()
            assert all(sha(path)==h for path,h in frozen.items())
            state.update(complete=True,phase='complete',ended_utc=datetime.now(timezone.utc).isoformat(),
                full_GPU_FP32_and_selected_CPU_FP64_references_passed=True);save()
    except BaseException as exc:
        import traceback
        state.update(phase='failed',error=repr(exc),traceback=traceback.format_exc(),ended_utc=datetime.now(timezone.utc).isoformat());save();raise
    finally:
        (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
        signal.alarm(0)
if __name__=='__main__':main()
