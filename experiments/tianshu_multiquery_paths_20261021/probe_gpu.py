# SPDX-License-Identifier: Apache-2.0
"""Unchanged byte-codec attention, 2D fallback vs small-query 3D; no model score."""
import argparse,gc,hashlib,importlib.util,json,os,signal,statistics,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from factory import BASE,OLD,PIPE,CONFIGS,sha,segments,create,DEPENDENCIES,EMULATE

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);args=p.parse_args()
    seconds=int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert seconds>600
    signal.alarm(min(2400,seconds))
    hardware=subprocess.check_output(['ixsmi'],text=True);assert 'Iluvatar BI-V150' in hardware and 'No running processes found' in hardware
    root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
    (root/'hardware_before.txt').write_text(hardware)
    import torch,vllm_fl
    emulator_spec=importlib.util.spec_from_file_location("_paths_short_BF16_P_emulation",EMULATE);emulator=importlib.util.module_from_spec(emulator_spec);emulator_spec.loader.exec_module(emulator)
    from vllm.platforms import current_platform
    assert current_platform.vendor_name=='iluvatar' and torch.cuda.device_count()==1
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    cpu=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert cpu['complete'] and cpu['numerical_JIT_AST_unchanged']
    frozen={**cpu['sources'],str(BASE/'CPU_GATE_32g.json'):sha(BASE/'CPU_GATE_32g.json')}
    for path,h in frozen.items():assert sha(path)==h
    planes_path=PIPE/'value_planes.py'
    spec=importlib.util.spec_from_file_location('_multiquery_value_plane',planes_path);planes=importlib.util.module_from_spec(spec);spec.loader.exec_module(planes)
    modes,generated,dense=create(root)
    frozen.update({str(p):sha(p) for p in generated})
    for path,h in frozen.items():
        target=root/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==h
    result=dict(complete=False,phase='component',runner_pid=os.getpid(),frozen_sources=frozen,source_count=len(frozen),
        raw_root=str(root),started_utc=datetime.now(timezone.utc).isoformat(),modes=[{k:v for k,v in m.items() if k in ('label','mode','config','proof')} for m in modes],
        torch_version=torch.__version__,GPU=torch.cuda.get_device_name(0),cases=[],timing_rounds=[],
        candidate_numerical_JIT_AST_unchanged=True,model_executed=False,official_benchmark=False,model_Math=False,adopted=False,
        timing_scope='One attention read only, 7 rotating rounds x20 hot CUDA Graph replays; shared KV fixture, no writer/model/sampler/draft costs; control includes actual gather when it fits.',
        baseline_scope='Actual bounded gather+native BF16, byte2D over capacity; mixed fitting fixtures are arithmetic references, no service speedup claim.')
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(root/'result.json')
    def quantize(x):
        bits=x.view(torch.int16).int();magnitude=bits&32767
        rounded=((magnitude+7+((magnitude>>4)&1))>>4)-928
        return (torch.where(rounded>=8,rounded.clamp(max=127),0)|((bits&32768)>>8)).to(torch.uint8)
    def decode(x):
        a=x.int();mag=a&127
        return (torch.where(mag>=8,(mag<<4)+14848,0)|((a&128)<<8)).to(torch.int16).view(torch.bfloat16)
    def err(a,b):
        d=(a.float()-b.float()).abs();bad=d>(.002+.01*b.float().abs())
        return dict(passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not bad.any()),max_abs=float(d.max()),failed=int(bad.sum()),atol=.002,rtol=.01)
    def ref(q,k,v,qlens,lengths,causal):
        out=[];offset=0
        for seq,(ql,kl) in enumerate(zip(qlens,lengths,strict=True)):
            query=q[offset:offset+ql].float().transpose(0,1)
            key=k[seq,:kl].float().repeat_interleave(8,dim=1).permute(1,2,0)
            value=v[seq,:kl].float().repeat_interleave(8,dim=1).transpose(0,1)
            score=(query@key)*128**-.5
            if causal:score.masked_fill_(torch.arange(kl,device=q.device)[None,None,:]>kl-ql+torch.arange(ql,device=q.device)[None,:,None],-float('inf'))
            out.append((score.softmax(-1)@value).transpose(0,1));offset+=ql
        return torch.cat(out)
    def measure(graph):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
        for _ in range(20):graph.replay()
        b.record();b.synchronize();return a.elapsed_time(b)/20
    cases=[(f'q{q}_n{n}_k{kl}',[q]*n,[kl]*n,False,True) for n in (1,8,32,64) for q in (2,3,4,8) for kl in (4096,)]
    cases += [('q8_n64_k5120',[8]*64,[5120]*64,False,True),('mixed_q_n8_strided',[1,2,3,4,7,8,2,3],[4097-i*73 for i in range(8)],True,True),
        ('mixed_q_n64_strided',[1,2,3,4,7,8,2,3]*8,[5120-i%7*83 for i in range(64)],True,True)]
    cases += [(f'q{q}_short_k{k}',[q]*8,[k]*8,True,True) for q,k in ((2,2),(3,17),(4,33),(8,8),(8,65))]
    cases += [('q7_n8_noncausal',[7]*8,[4097]*8,True,False),('q7_n64_noncausal',[7]*64,[4096]*64,False,False)]
    save()
    try:
      with torch.inference_mode():
        for case_id,(name,qlens,lengths,strided,causal) in enumerate(cases):
            generator=torch.Generator(device='cuda').manual_seed(202610219000+case_id)
            n=len(qlens);tokens=sum(qlens);width=(max(lengths)+15)//16;pages=n*width;s=segments(n)
            klog=torch.randn(n,width*16,2,128,device='cuda',dtype=torch.bfloat16,generator=generator)
            vlog=torch.randn(klog.shape,device='cuda',dtype=torch.bfloat16,generator=generator);qstore=torch.randn(tokens,20 if strided else 16,128,device='cuda',dtype=torch.bfloat16,generator=generator);q=qstore[:,:16]
            cu=torch.tensor([0,*__import__('itertools').accumulate(qlens)],device='cuda',dtype=torch.int32)
            used=torch.tensor(lengths,device='cuda',dtype=torch.int32)
            tabstore=torch.full((n,width+9),-2000000,device='cuda',dtype=torch.int32);table=tabstore[:,:width]
            physical_pages=(pages+2)*(2 if strided else 1);raw=torch.full((physical_pages*12288,),17,device='cuda',dtype=torch.uint8)
            full=torch.as_strided(raw,(physical_pages,2,16,2,128),(12288,4096,256,128,1));view=full[::2] if strided else full
            k,v=view.unbind(1);v=planes.value_plane(v)
            fixtures=[]
            for m in modes:
                outstore=torch.full((tokens+13,20 if strided else 16,128),17.,device='cuda',dtype=torch.bfloat16);out=outstore[:tokens,:16]
                scratch=torch.full((tokens+1,16,s,128),17.,device='cuda',dtype=torch.float32)
                maxima=torch.full((tokens+1,16,s),17.,device='cuda',dtype=torch.float32);expsum=torch.full_like(maxima,17.)
                kw=dict(q=q,k=k,v=v,out=out,cu_seqlens_q=cu,max_seqlen_q=max(qlens),seqused_k=used,max_seqlen_k=width*16,
                    softmax_scale=128**-.5,causal=causal,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=None,k_descale=None,v_descale=None,
                    seq_threshold_3D=64,num_par_softmax_segments=s,softmax_segm_output=scratch,softmax_segm_max=maxima,softmax_segm_expsum=expsum)
                fixtures.append(dict(mode=m,kw=kw,outstore=outstore,out=out,scratch=scratch,maxima=maxima,expsum=expsum,graph=None))
            row=dict(name=name,sequences=n,query_lengths=qlens,key_lengths=lengths,strided=strided,causal=causal,
                physical_page_bytes=12288,physical_stride_bytes=k.stride(0),scratch_per_mode_bytes=(tokens+1)*16*s*130*4,
                full_prefill_arena_fits_8192_pages=pages<=8192,baseline_matches_uniform_or_fallback_path=(all(q>1 for q in qlens) or pages>8192) and causal,phases=[],timings={})
            result['cases'].append(row);save();archive=dict(query_lengths=qlens,phases=[])
            for phase in range(2):
                if phase:
                    q.neg_();klog.mul_(.5);vlog.neg_()
                    lengths=[max(ql,kl-min(7,kl-ql)) for ql,kl in zip(qlens,lengths,strict=True)];used.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
                table.copy_(torch.randperm(pages,device='cuda',generator=generator).int().reshape(n,width))
                encoded=quantize(klog);decoded=decode(encoded);ids=table.flatten().long()
                k[ids]=encoded.reshape(pages,16,2,128);v[ids]=vlog.reshape(pages,16,2,128)
                raw_before=raw.clone();q_before=qstore.clone();cu_before=cu.clone();table_before=tabstore.clone();used_before=used.clone()
                expected=ref(q,decoded,vlog,qlens,lengths,causal);torch.cuda.synchronize()
                cpu_checks=[];selected=[];offset=0
                for seq,(ql,kl) in enumerate(zip(qlens,lengths,strict=True)):
                    if seq in {0,n//2,n-1}:
                        kc=decoded[seq,:kl].cpu().float();vc=vlog[seq,:kl].cpu().float();qc=q[offset:offset+ql].cpu().float()
                        for pos in sorted({0,ql-1}):
                          for head in (0,7,8,15):
                            end=kl-ql+pos+1 if causal else kl
                            answer=((kc[:end,head//8]@qc[pos,head])*128**-.5).softmax(0)@vc[:end,head//8]
                            check=err(expected[offset+pos,head].cpu(),answer);assert check['passed'];cpu_checks.append(check)
                            selected.append(dict(token=offset+pos,head=head,seq=seq,pos=pos))
                    offset+=ql
                checks=[];outputs={};native_references={}
                for f in fixtures:
                    m=f['mode'];cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
                    if phase==0:
                        for _ in range(2):m['reader'].unified_attention(**f['kw'])
                        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):m['reader'].unified_attention(**f['kw'])
                        f['graph']=g
                    f['graph'].replay();torch.cuda.synchronize()
                    ideal_check=err(f['out'],expected)
                    if max(lengths)<=65:
                        config=m['config'] or dict(block_m=256,tile=128,warps=16,stages=1)
                        reference=emulator.emulate(q.cpu(),decoded.cpu(),vlog.cpu(),qlens,lengths,config,m['mode']=='candidate',s,causal)
                        check=err(f['out'].cpu(),reference);native_references[m['label']]=reference
                        check.update(reference_kind='independent_CPU_native_BF16_P',ideal_FP32_diagnostic=ideal_check)
                    else:
                        check=ideal_check;check.update(reference_kind='ideal_FP32')
                    assert check['passed'],(name,m['label'],phase,check)
                    assert torch.equal(raw,raw_before) and torch.equal(qstore,q_before) and torch.equal(cu,cu_before) and torch.equal(tabstore,table_before) and torch.equal(used,used_before)
                    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
                    assert bool((f['outstore'][tokens:]==17).all())
                    if strided:assert bool((f['outstore'][:tokens,16:]==17).all())
                    assert bool((f['scratch'][-1]==17).all() and (f['maxima'][-1]==17).all() and (f['expsum'][-1]==17).all())
                    check.update(label=m['label'],full_output_reference_checked=True,input_KV_and_metadata_unchanged=True,RNG_unchanged=True,guards_preserved=True,
                        actual_producer=m['producer'].last,actual_reducer=m['reducer'].last)
                    checks.append(check);outputs[m['label']]=f['out'].cpu()
                archive['phases'].append(dict(phase=phase,key_lengths=lengths,selected=selected,q=q.cpu(),
                    selected_original_K={seq:klog[seq,:lengths[seq]].cpu() for seq in sorted({0,n//2,n-1})},selected_K={seq:decoded[seq,:lengths[seq]].cpu() for seq in sorted({0,n//2,n-1})},
                    selected_V={seq:vlog[seq,:lengths[seq]].cpu() for seq in sorted({0,n//2,n-1})},expected=expected.cpu(),native_BF16_P_references=native_references,short_fixture=(dict(original_K=klog.cpu(),decoded_K=decoded.cpu(),V=vlog.cpu()) if max(lengths)<=65 else None),outputs=outputs))
                row['phases'].append(dict(phase=phase,key_lengths=lengths,full_GPU_checks=checks,CPU_checks=cpu_checks));save()
                del expected,raw_before,q_before,cu_before,table_before,used_before
            archive_path=root/(name+'.pt');torch.save(archive,archive_path);row['tensor_archive_sha256']=sha(archive_path)
            timed_raw_before=raw.clone();timed_q_before=qstore.clone();timed_table_before=tabstore.clone();timed_used_before=used.clone();timed_cpu_rng=torch.get_rng_state().clone();timed_gpu_rng=torch.cuda.get_rng_state().clone()
            for round_index in range(7):
                shift=round_index%len(fixtures);order=fixtures[shift:]+fixtures[:shift]
                for f in order:
                    elapsed=measure(f['graph']);result['timing_rounds'].append(dict(case=name,round=round_index,label=f['mode']['label'],ms=elapsed,replays=20))
            for f in fixtures:
                label=f['mode']['label'];times=[r['ms'] for r in result['timing_rounds'] if r['case']==name and r['label']==label]
                row['timings'][label]=dict(median_ms=statistics.median(times),all_ms=times)
                assert bool((f['outstore'][tokens:]==17).all() and (f['scratch'][-1]==17).all() and (f['maxima'][-1]==17).all() and (f['expsum'][-1]==17).all())
            assert torch.equal(raw,timed_raw_before) and torch.equal(qstore,timed_q_before) and torch.equal(tabstore,timed_table_before) and torch.equal(used,timed_used_before)
            assert torch.equal(timed_cpu_rng,torch.get_rng_state()) and torch.equal(timed_gpu_rng,torch.cuda.get_rng_state())
            row["timed_full_KV_inputs_RNG_preserved"]=True
            assert bool((dense.storage[-1]==17).all()) and torch.equal(dense.dense_ids,torch.arange(8192,device=q.device,dtype=torch.int32))
            row["timed_native_arena_guard_and_ids_preserved"]=True
            del timed_raw_before,timed_q_before,timed_table_before,timed_used_before
            baseline=row['timings']['bounded_gather_or_byte']['median_ms'];best=min((m for m in row['timings'] if m!='bounded_gather_or_byte'),key=lambda m:row['timings'][m]['median_ms'])
            row.update(best_exploratory_label=best,best_exploratory_gain_percent=(baseline/row['timings'][best]['median_ms']-1)*100)
            save();print(json.dumps(dict(case=name,baseline_ms=baseline,best_label=best,best_ms=row['timings'][best]['median_ms'],gain_percent=row['best_exploratory_gain_percent'])),flush=True)
            for f in fixtures:f['graph'].reset()
            del fixtures,archive,outputs,klog,vlog,qstore,q,raw,full,view,k,v,decoded,encoded,table,tabstore,used,cu,scratch,maxima,expsum,out,outstore,f,g,kw
            gc.collect();torch.cuda.empty_cache()
        assert len(result['cases'])==26 and all(sha(p)==h for p,h in frozen.items())
        result.update(complete=True,phase='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()
    except BaseException as e:
        result.update(phase='failed',error=repr(e));save();raise
if __name__=='__main__':main()
