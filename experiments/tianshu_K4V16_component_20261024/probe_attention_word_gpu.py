# SPDX-License-Identifier: Apache-2.0
"""Complete writer+actual paged attention, all outputs, same Graph changed data."""
import argparse,hashlib,importlib.util,json,os,signal,statistics,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
CONTRACT=BASE.parent/'tianshu_K4V16_contract_20261024'
PARENT=Path('/workspace/benchmark-results/20261024_W4_QKVO64_fitted_official4k_tianshu/pair_load_records/generated_reader/generated_pair_rows.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.output.resolve();assert root.is_relative_to('/dev/shm') and not root.exists();root.mkdir()
    remaining=int((datetime.fromisoformat('2026-10-24T15:58:00+00:00')-datetime.now(timezone.utc)).total_seconds())
    assert remaining>900;signal.alarm(min(3600,remaining))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware
    (root/'hardware_before.txt').write_text(hardware)
    gate=json.loads((BASE/'GPU_WRITER_GATE_32g.json').read_text());assert gate['complete'] and len(gate['cases'])==4
    for path,h in gate['frozen_sources'].items():assert sha(path)==h
    import torch
    import vllm_fl
    import component_launch as launch
    import fixtures_cpu as fixtures
    import k4_word_reader as k4_reader
    import pages,writer,word_decode_gpu
    from vllm.v1.kv_cache_interface import KVQuantMode
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    codec=load('_K4_component_CPU_codec',CONTRACT/'codec.py')
    current_writer_path=BASE.parent/'tianshu_soft8_mixed_planes_20261019/writer.py'
    current_writer=load('_K4_current_writer',current_writer_path)
    current=load('_K4_current_f1df_reader',PARENT)
    candidate,proof=k4_reader.load(PARENT,root/'candidate_reader')
    decoder_primitive=word_decode_gpu.verify(candidate,root/'decode_primitive')
    assert sha(PARENT)==k4_reader.PARENT_SHA
    models=[]
    for label,reader in [('current',current),('k4',candidate)]:
        invoke,producer,observed=launch.attach(reader,label,root)
        models.append(dict(label=label,invoke=invoke,producer=producer,observed=observed))
    sources=list(BASE.glob('*.py'))+[BASE/'PLAN_32g.md',BASE/'WORD_PLAN_32g.md',BASE/'GPU_WRITER_GATE_32g.json',PARENT,Path(candidate.__file__),
        current_writer_path,launch.NATIVE,launch.native.LEGACY,launch.native.POLICY,CONTRACT/'codec.py',CONTRACT/'CPU_GATE_32g.json']
    frozen={str(f):sha(f) for f in sources}
    for f in sources:
        copied=root/'source_snapshot'/f.relative_to('/');copied.parent.mkdir(parents=True,exist_ok=True);copied.write_bytes(f.read_bytes())
    result=dict(complete=False,phase='K4 paired decode component',runner_pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
        frozen_sources=frozen,reader_proof=proof,launch_proof=launch.proof(),decoder_primitive=decoder_primitive,cases=[],numpy_version=np.__version__,torch_version=torch.__version__,
        native_spec=pages.native_spec(),control='actual r5 f1df paired software K8/BF16 V',candidate='per-token/head K4/BF16 V',
        current_model_not_changed=True,model_executed=False,engine_allocator_connected=False,official_benchmark=False,adopted=False,
        time_scope='one KV update + one paged decode layer; hot Graph, alternating 9 trials x20 replays',
        resident33_scope='standalone unpadded C33 component; production r5 usually graph-pads actual33 to40',
        FP64_all_output_verification_pending=True)
    def save():
        t=root/'result.tmp';t.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');t.replace(root/'result.json')
    def gpu(x):return torch.from_numpy(np.ascontiguousarray(x)).to('cuda')
    def bf16(x):return gpu(np.ascontiguousarray(x).view(np.int16)).view(torch.bfloat16)
    def err(actual,expected):
        a=actual.float();e=expected.float();d=(a-e).abs();bad=d>.002+.01*e.abs()
        return dict(passed=bool(torch.isfinite(a).all() and torch.isfinite(e).all() and not bad.any()),
            max_abs=float(d.max()),failed_elements=int(bad.sum()),atol=.002,rtol=.01,elements=a.numel())
    def reference(q,k,v,used,scales=None):
        n,width=k.shape[:2];query=q.float().reshape(n,2,8,128)
        scores=torch.einsum('bhgd,bkhd->bhgk',query,k.float())
        scores=scores*(128**-.5) if scales is None else scores*(scales.transpose(1,2)*(128**-.5))[:,:,None,:]
        mask=torch.arange(width,device='cuda')[None,:]<used[:,None]
        scores.masked_fill_(~mask[:,None,None,:],-float('inf'))
        return torch.einsum('bhgk,bkhd->bhgd',scores.softmax(-1),v.float()).reshape(n,16,128)
    save()
    try:
        with torch.inference_mode():
            for case_id,(name,base_lengths,strided,transition) in enumerate(fixtures.CASES):
                seed=2026102400+case_id;n=len(base_lengths);width=(max(base_lengths)+15)//16;count=n*width
                q_storage=torch.full((n+13,20 if strided else 16,128),17.,device='cuda',dtype=torch.bfloat16)
                q=q_storage[:n,:16]
                table_storage=torch.full((n,width+3 if strided else width),17,device='cuda',dtype=torch.int32);table=table_storage[:,:width]
                used_storage=torch.full((n+3,),17,device='cuda',dtype=torch.int32);used=used_storage[:n]
                cu=torch.arange(n+1,device='cuda',dtype=torch.int32)
                fs=[]
                for model in models:
                    label=model['label'];page_bytes=12288 if label=='current' else pages.PAGE_BYTES
                    physical=(count+2)*(2 if strided else 1)
                    raw=torch.full((physical*page_bytes,),17,device='cuda',dtype=torch.uint8)
                    if label=='k4':k,v,s=pages.views(raw,physical,strided)
                    else:
                        k=torch.as_strided(raw,(physical,16,2,128),(12288,256,128,1))
                        v=torch.empty(0,device='cuda',dtype=torch.bfloat16).set_(raw.untyped_storage(),2048,(physical,16,2,128),(6144,256,128,1))
                        if strided:k,v=k[::2],v[::2]
                        s=None
                    out_storage=torch.full_like(q_storage,17.);out=out_storage[:n,:16]
                    key_storage=torch.full((n+3,3,128),19.,device='cuda',dtype=torch.bfloat16)
                    value_storage=torch.full_like(key_storage,23.)
                    slots=torch.full((n+3,),-1,device='cuda',dtype=torch.int64)
                    kw=dict(q=q,k=k,v=v,out=out,cu_seqlens_q=cu,max_seqlen_q=1,seqused_k=used,
                        max_seqlen_k=1 if transition else width*16,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),
                        block_table=table,softcap=0,q_descale=None,k_descale=torch.ones(1,device='cuda').expand(n,2),
                        v_descale=torch.ones(1,device='cuda').expand(n,2),seq_threshold_3D=64,num_par_softmax_segments=64,
                        softmax_segm_output=torch.full((64,16,64,128),17.,device='cuda'),
                        softmax_segm_max=torch.full((64,16,64),17.,device='cuda'),softmax_segm_expsum=torch.full((64,16,64),17.,device='cuda'),
                        kv_quant_mode=KVQuantMode.NONE if label=='current' else KVQuantMode.INT8_PER_TOKEN_HEAD,
                        k_scale_cache=s,v_scale_cache=s)
                    fs.append(dict(model=model,label=label,raw=raw,k=k,v=v,s=s,kw=kw,page_bytes=page_bytes,
                        physical_pages=physical,out_storage=out_storage,out=out,key_storage=key_storage,value_storage=value_storage,
                        key=key_storage[:,:2],value=value_storage[:,:2],slots=slots,graphs=None))
                phase_records=[]
                for phase in range(2):
                    kb,vb,qb,tb=fixtures.generate(seed,n,width,phase)
                    lengths=([1]*n if transition and phase==0 else list(base_lengths) if transition or not phase
                        else [x-min(7,x-1) for x in base_lengths])
                    q.copy_(bf16(qb));table.copy_(gpu(tb));used.copy_(gpu(np.asarray(lengths,np.int32)))
                    encoded=fixtures.k8_encode(kb);packed,scales=codec.encode(fixtures.values(kb))
                    positions=np.asarray(lengths,np.int64)-1
                    slot_ids=tb[np.arange(n),positions//16].astype(np.int64)*16+positions%16
                    slot_ids=np.concatenate([slot_ids,np.full(3,-1,np.int64)])
                    raw_key=kb[np.arange(n),positions];raw_value=vb[np.arange(n),positions]
                    ids=gpu(tb.reshape(-1).astype(np.int64))
                    v_float=gpu(fixtures.values(vb))
                    original_ref=reference(q,gpu(fixtures.values(kb)),v_float,used)
                    shared_inputs=[q_storage,table_storage,used_storage,cu]
                    shared_before=[x.clone() for x in shared_inputs]
                    checks=[];artifacts=dict(lengths=np.asarray(lengths,np.int32),table=tb)
                    for f in fs:
                        label=f['label'];f['raw'].fill_(17)
                        f['k'][ids]=gpu((encoded if label=='current' else packed).reshape(count,16,2,-1))
                        f['v'][ids]=bf16(vb.reshape(count,16,2,128))
                        if label=='k4':f['s'][ids]=gpu(scales.reshape(count,16,2))
                        expected_raw=f['raw'].clone()
                        f['key'][:n].copy_(bf16(raw_key));f['value'][:n].copy_(bf16(raw_value));f['slots'].copy_(gpu(slot_ids))
                        tail_slots=f['slots'][:n]
                        f['k'][tail_slots//16,tail_slots%16]=255;f['v'][tail_slots//16,tail_slots%16]=-13.
                        if label=='k4':f['s'][tail_slots//16,tail_slots%16]=-13.
                        for field in ('softmax_segm_output','softmax_segm_max','softmax_segm_expsum'):f['kw'][field].fill_(17.)
                        f['out_storage'].fill_(17.)
                        input_tensors=[f['key_storage'],f['value_storage'],f['slots']]
                        input_before=[x.clone() for x in input_tensors]
                        def update(f=f):
                            return (current_writer.write(f['key'],f['value'],f['k'],f['v'],f['slots']) if f['label']=='current'
                                else writer.write(f['key'],f['value'],f['k'],f['v'],f['s'],f['slots']))
                        def read(f=f):return f['model']['invoke'](f['kw'])
                        def complete(update=update,read=read):update();read()
                        if f['graphs'] is None:
                            compiled_writer=update();read();torch.cuda.synchronize()
                            f['writer_binary_sha256']=hashlib.sha256(compiled_writer.kernel).hexdigest()
                            f['graphs']={}
                            for scope,fn in [('full',complete),('reader',read),('writer',update)]:
                                graph=torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):fn()
                                f['graphs'][scope]=graph
                        rng=torch.cuda.get_rng_state().clone();f['graphs']['full'].replay();torch.cuda.synchronize()
                        assert torch.equal(torch.cuda.get_rng_state(),rng)
                        assert torch.equal(f['raw'],expected_raw),'complete raw cache mismatch'
                        assert bool((f['out_storage'][n:]==17).all())
                        if strided:assert bool((f['out_storage'][:n,16:]==17).all())
                        for x,before in zip(shared_inputs+input_tensors,shared_before+input_before):
                            assert torch.equal(x.view(torch.int16) if x.dtype==torch.bfloat16 else x,before.view(torch.int16) if before.dtype==torch.bfloat16 else before)
                        segments=launch.configuration(n,f['kw']['max_seqlen_k'])['segments']
                        for field,factor in [('softmax_segm_output',128),('softmax_segm_max',1),('softmax_segm_expsum',1)]:
                            assert bool((f['kw'][field].flatten()[n*16*segments*factor:]==17).all())
                        if label=='current':kref=gpu(fixtures.values(fixtures.k8_decode(encoded)));expected=reference(q,kref,v_float,used)
                        else:kref=gpu(codec.unpack(packed));expected=reference(q,kref,v_float,used,gpu(scales))
                        numerical=err(f['out'],expected);assert numerical['passed'],(name,phase,label,numerical)
                        approximation=float(torch.linalg.vector_norm(expected-original_ref)/torch.linalg.vector_norm(original_ref))
                        extra={}
                        if label=='k4':
                            bf16_dequantized=(kref.float()*gpu(scales)[...,None]).to(torch.bfloat16)
                            alternate=reference(q,bf16_dequantized,v_float,used)
                            extra['postscale_vs_BF16_prefill_reconstruction_max_abs']=float((alternate-expected).abs().max())
                        raw_hash=fixtures.digest(f['raw'].cpu().numpy())
                        checks.append(dict(label=label,numerical=numerical,all_raw_bytes_equal=True,raw_storage_sha256=raw_hash,
                            raw_cache_bytes=f['raw'].numel(),physical_pages=f['physical_pages'],page_bytes=f['page_bytes'],
                            writer_binary_sha256=f['writer_binary_sha256'],relative_L2_to_unquantized_reference=approximation,
                            output_and_segment_guards=True,input_padding_and_slots_unchanged=True,RNG_unchanged=True,**extra))
                        artifacts[label+'_output_bits']=f['out'].cpu().view(torch.int16).numpy().view(np.uint16)
                        artifacts[label+'_GPU_reference']=expected.cpu().numpy()
                    artifact=root/f'{name}_phase{phase}.npz';np.savez_compressed(artifact,**artifacts)
                    record=dict(phase=phase,lengths=lengths,checks=checks,same_Graph_changed_Q_K_V_table_lengths_slots=bool(phase),
                        input_hashes=dict(K_bits=fixtures.digest(kb),V_bits=fixtures.digest(vb),Q_bits=fixtures.digest(qb),table=fixtures.digest(tb)),
                        artifact=dict(path=str(artifact),sha256=sha(artifact),bytes=artifact.stat().st_size))
                    phase_records.append(record)
                    del expected_raw,kref,expected,original_ref,v_float,kb,vb,qb,encoded,packed,scales
                torch.cuda.synchronize();time.sleep(.05)
                measurements={scope:[[],[]] for scope in ('full','reader','writer')}
                for scope in measurements:
                    for trial in range(9):
                        for i in (range(2) if trial%2==0 else (1,0)):
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(20):fs[i]['graphs'][scope].replay()
                            end.record();end.synchronize();measurements[scope][i].append(start.elapsed_time(end)/20)
                medians={scope:list(map(statistics.median,values)) for scope,values in measurements.items()}
                entry=dict(name=name,seed=seed,batch=n,width=width,strided=strided,capture_transition=transition,
                    phases=phase_records,raw_ms=measurements,median_ms=medians,
                    full_component_speedup_percent=100*(medians['full'][0]/medians['full'][1]-1),
                    actual_launch={m['label']:m['observed'].last for m in models})
                result['cases'].append(entry);save()
                print(json.dumps({k:entry[k] for k in ('name','median_ms','full_component_speedup_percent')}),flush=True)
                for f in fs:f['graphs'].clear()
                del fs,shared_inputs,shared_before,input_tensors,input_before,q_storage,table_storage,used_storage,cu,ids
                torch.cuda.empty_cache()
        for path,h in frozen.items():assert sha(path)==h
        result.update(complete=True,phase='GPU component complete pending independent FP64 verification',ended_at=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as e:
        result.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
    finally:
        signal.alarm(0);(root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
