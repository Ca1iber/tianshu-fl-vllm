# SPDX-License-Identifier: Apache-2.0
"""K-only page packing, changed-data Graph writer/attention and controlled events."""
import argparse,hashlib,importlib.util,itertools,json,os,runpy,signal,statistics,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from source import ATTENTION_SHA,WRITER_SHA,CHUNKS,render_attention,render_writer


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);a=p.parse_args()
    root=a.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
    remaining=int((datetime.fromisoformat(a.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>900
    signal.alarm(min(remaining,1800));before=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in before
    (root/'hardware_before.txt').write_text(before)
    base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent/'tianshu_gemm_topp_geo_20261016'))
    from validation_profile_new import prepare
    command,env,profile,frozen=prepare(root,Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
    assert len(profile)==38;os.environ.clear();os.environ.update(env)
    runpy.run_path(str(Path(profile['PYTHONPATH'])/'sitecustomize.py'))
    import torch,vllm_fl
    import vllm.v1.attention.ops.triton_unified_attention as native
    import vllm.v1.attention.ops.triton_reshape_and_cache_flash as writer
    from vllm.v1.kv_cache_interface import KVQuantMode
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    raw_attention=Path(sys.modules['tianshu_native_cache_policy'].__file__).read_bytes()
    raw_writer=Path(writer.__file__).read_bytes()
    assert hashlib.sha256(raw_attention).hexdigest()==ATTENTION_SHA
    assert hashlib.sha256(raw_writer).hexdigest()==WRITER_SHA
    gate=json.loads((base/'CPU_GATE_V2_32g.json').read_text());assert gate['complete'] and gate['physical_address_cases']==16384
    for name,sha in gate['sources'].items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==sha
    for path in (Path(__file__),base/'source.py',base/'check_cpu_v2.py',base/'run_probe_v2.py',base/'CPU_GATE_V2_32g.json',Path(native.__file__),Path(writer.__file__)):
        frozen[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name,sha in frozen.items():
        dst=root/'source_snapshot'/Path(name).relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(name).read_bytes());assert hashlib.sha256(dst.read_bytes()).hexdigest()==sha
    data=dict(phase='preparing',complete=False,runner_pid=os.getpid(),frozen_sources=frozen,generated_sources={},profile_environment=profile,server_command=command,cases=[],events=[],official_benchmark=False,adopted=False,
        scope='Owned writer and attention components; no live model, actual scheduler/backend dispatch, throughput or quality claim',
        same_CV_L0_attention_arithmetic=True,V_layout_unchanged=True,page16_BF16=True)
    def save():
        t=root/'result.tmp';t.write_text(json.dumps(data,indent=2)+'\n');t.replace(root/'result.json')
    def load(text,name):
        f=root/'generated'/f'{name}.py';f.parent.mkdir(parents=True,exist_ok=True);f.write_text(text)
        spec=importlib.util.spec_from_file_location(name,f);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
        data['generated_sources'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();return m
    class Producer:
        def __init__(self,kernel):self.kernel=kernel;self.tile=None;self.metadata=[]
        def __getitem__(self,grid):
            def launch(*args,**kw):
                launch_grid=grid
                assert not args and kw['BLOCK_SIZE']==16 and kw['HEAD_SIZE']==128
                assert kw['num_query_heads']==16 and kw['num_queries_per_kv']==8
                assert kw['KV_QUANT_MODE']==0 and not kw['USE_TD'] and not kw['USE_TD_QO']
                cache=kw['key_cache_ptr']
                assert tuple(cache.shape[1:])==(16,2,128) and cache.stride(1)==256 and cache.stride(2)==128 and cache.stride(3)==1 and cache.stride(0)>=4096
                assert kw['query_ptr'].dtype==kw['key_cache_ptr'].dtype==kw['value_cache_ptr'].dtype==torch.bfloat16
                if kw['IS_3D']:
                    n=kw['num_seqs'];selected=n in (8,16,32,40,48,56,64)
                    tile=16 if selected else 32;warps=1 if selected else 2 if 8<=n<=64 else 4
                    assert kw['BLOCK_M']==16 and kw['BLOCK_Q']==2
                    kw.update(TILE_SIZE=tile,num_warps=warps,num_stages=1)
                else:
                    kw.update(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=128,num_warps=16,num_stages=1)
                    launch_grid=(kw['query_ptr'].shape[0]//32+kw['num_seqs'],grid[1])
                kw['KV_CACHE_MODIFIER']='.cv';self.tile=kw['TILE_SIZE']
                result=self.kernel[launch_grid](**kw)
                self.metadata.append(dict(IS_3D=kw['IS_3D'],tile=self.tile,grid=list(launch_grid),num_warps=result.metadata.num_warps,num_stages=result.metadata.num_stages,segments=kw['NUM_SEGMENTS_PER_SEQ'],n_regs=getattr(result,'n_regs',None),n_spills=getattr(result,'n_spills',None)))
                return result
            return launch
    class Reducer:
        def __init__(self,kernel,producer):self.kernel=kernel;self.producer=producer
        def __getitem__(self,grid):
            def launch(*args,**kw):kw['TILE_SIZE']=self.producer.tile;return self.kernel[grid](*args,**kw)
            return launch
    modules={}
    for x in (0,*CHUNKS):
        am=load(raw_attention.decode() if not x else render_attention(raw_attention,x),f'_tianshu_kblock_attention_{x}')
        wm=load(raw_writer.decode() if not x else render_writer(raw_writer,x),f'_tianshu_kblock_writer_{x}')
        producer=Producer(am.kernel_unified_attention);am.kernel_unified_attention=producer;am.reduce_segments=Reducer(am.reduce_segments,producer)
        modules[x]=(am,wm,producer)
    def err(actual,expected):
        actual,expected=actual.float(),expected.float();d=(actual-expected).abs();bad=d>.02+.01*expected.abs()
        return dict(passed=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all() and not bad.any()),max_abs=float(d.max()),bad=int(bad.sum()),atol=.02,rtol=.01)
    def reference(q,k,v,qlens,lengths):
        outputs=[];begin=0
        for i,(ql,kl) in enumerate(zip(qlens,lengths)):
            keys=k[i,:kl].float().repeat_interleave(8,1);values=v[i,:kl].float().repeat_interleave(8,1);query=q[begin:begin+ql].float().transpose(0,1);parts=[]
            for off in range(0,ql,64):
                chunk=query[:,off:off+64];scores=(chunk@keys.permute(1,2,0))*128**-.5
                allowed=torch.arange(kl,device=q.device)[None,:]<=kl-ql+torch.arange(off,off+chunk.shape[1],device=q.device)[:,None]
                scores.masked_fill_(~allowed[None],-float('inf'));parts.append(scores.softmax(-1)@values.transpose(0,1))
            outputs.append(torch.cat(parts,1).transpose(0,1));begin+=ql
        return torch.cat(outputs)
    cases=[('decode64_4k',[1]*64,[4096]*64,False),('decode64_5k',[1]*64,[5120]*64,True),
        ('decode32_ragged',[1]*32,[4096+i%16 for i in range(32)],True),('decode1_33',[1],[33],True),
        ('prefill33',[33],[65],True),('prefill2048',[2048],[2048],False),('chunk2048',[2048],[4096],True),
        ('ragged_prefill',[31,32,33],[33,65,4096],True)]
    save()
    try:
        with torch.inference_mode():
            for case_id,(name,qlens,lengths,strided) in enumerate(cases):
                torch.manual_seed(2026101940+case_id);n=len(qlens);total=sum(qlens);width=(max(lengths)+15)//16;pages=n*width
                logical_k=torch.randn(n,width*16,2,128,device='cuda',dtype=torch.bfloat16);logical_v=torch.randn_like(logical_k)
                q_storage=torch.randn(total,20 if strided else 16,128,device='cuda',dtype=torch.bfloat16);q=q_storage[:,:16]
                table=torch.randperm(pages,device='cuda').int().reshape(n,width)
                cu=torch.tensor([0]+list(itertools.accumulate(qlens)),device='cuda',dtype=torch.int32)
                used=torch.tensor(lengths,device='cuda',dtype=torch.int32)
                segs=8 if n==64 else 64 if n==1 else 4 if 16<=n<=64 else 16
                fixtures=[]
                for x in (0,*CHUNKS):
                    raw_pages=(pages+2)*(2 if strided else 1)
                    storage=torch.full((raw_pages,2,16,2,128),17.,device='cuda',dtype=q.dtype)
                    kv=storage[::2] if strided else storage;k,v=kv.unbind(1)
                    packed=k if not x else k.as_strided((kv.shape[0],2,128//x,16,x),(k.stride(0),2048,16*x,x,1))
                    out_storage=torch.full((total+13,20 if strided else 16,128),17.,device='cuda',dtype=q.dtype);out=out_storage[:total,:16]
                    buffers=(torch.full((64,16,64,128),17.,device='cuda',dtype=torch.float32),
                             torch.full((64,16,64),17.,device='cuda',dtype=torch.float32),torch.full((64,16,64),17.,device='cuda',dtype=torch.float32))
                    fixtures.append(dict(x=x,storage=storage,kv=kv,k=k,v=v,packed=packed,out_storage=out_storage,out=out,buffers=buffers))
                def update_rows():
                    slots=[];ks=[];vs=[]
                    for i,(ql,kl) in enumerate(zip(qlens,lengths)):
                        pos=torch.arange(kl-ql,kl,device='cuda',dtype=torch.int64)
                        slots.append(table[i,pos//16].long()*16+pos%16);ks.append(logical_k[i,kl-ql:kl]);vs.append(logical_v[i,kl-ql:kl])
                    return (torch.cat((*slots,torch.full((3,),-1,device='cuda',dtype=torch.int64))),
                            torch.cat((*ks,torch.full((3,2,128),19.,device='cuda',dtype=q.dtype))),
                            torch.cat((*vs,torch.full((3,2,128),23.,device='cuda',dtype=q.dtype))))
                def invoke(f):
                    am,wm,producer=modules[f['x']]
                    assert tuple(f['k'].shape[1:])==(16,2,128) and f['k'].stride(1)==256 and f['k'].stride(2)==128 and f['k'].stride(3)==1
                    assert f['k'].stride()==f['v'].stride() and f['v'].data_ptr()-f['k'].data_ptr()==4096*2
                    wm.triton_reshape_and_cache_flash(f['ki'],f['vi'],f['k'],f['v'],f['slots'],'auto',1.,1.)
                    am.unified_attention(q=q,k=f['k'],v=f['v'],out=f['out'],cu_seqlens_q=cu,max_seqlen_q=max(qlens),
                        seqused_k=used,max_seqlen_k=max(lengths),softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=table,
                        softcap=0.,q_descale=None,k_descale=None,v_descale=None,seq_threshold_3D=64,num_par_softmax_segments=segs,
                        softmax_segm_output=f['buffers'][0],softmax_segm_max=f['buffers'][1],softmax_segm_expsum=f['buffers'][2],kv_quant_mode=KVQuantMode.NONE,use_td=False)
                phases=[]
                for phase in (0,1):
                    if phase:
                        logical_k.mul_(.5);logical_v.neg_();q.neg_()
                        lengths=[kl-min(7,max(0,kl-ql)) for ql,kl in zip(qlens,lengths)]
                        used.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
                    slots,ki,vi=update_rows();valid=slots[slots>=0];blocks=valid//16;tokens=valid%16
                    for f in fixtures:
                        x=f['x'];physical_k=logical_k.reshape(pages,16,2,128)
                        if x:physical_k=physical_k.reshape(pages,16,2,128//x,x).permute(0,2,3,1,4)
                        f['packed'][table.flatten().long()]=physical_k
                        f['v'][table.flatten().long()]=logical_v.reshape(pages,16,2,128)
                        expected=f['storage'].clone()
                        if not phase:f.update(slots=slots.clone(),ki=ki.clone(),vi=vi.clone())
                        else:f['slots'].copy_(slots);f['ki'].copy_(ki);f['vi'].copy_(vi)
                        if x:f['packed'][blocks,:,:,tokens,:]=-13.
                        else:f['k'][blocks,tokens]=-13.
                        f['v'][blocks,tokens]=-13.
                        if not phase:
                            for _ in range(3):invoke(f)
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):invoke(f)
                            f['graph']=graph
                        rng_before=torch.cuda.get_rng_state().clone()
                        f['graph'].replay();torch.cuda.synchronize()
                        assert torch.equal(torch.cuda.get_rng_state(),rng_before)
                        decodes=n if max(qlens)==1 else 0
                        for buf,scale in zip(f['buffers'],(128,1,1)):
                            used_count=decodes*16*segs*scale
                            assert bool((buf.flatten()[used_count:]==17.).all())
                        assert torch.equal(f['storage'],expected),(name,x,'writer physical storage')
                        assert bool((f['out_storage'][total:]==17.).all())
                        if strided:assert bool((f['out_storage'][:total,16:]==17.).all())
                        f['expected_storage']=expected
                    expected=reference(q,logical_k,logical_v,qlens,lengths);cpu=[];begin=0
                    for i,(ql,kl) in enumerate(zip(qlens,lengths)):
                        if i in (0,n//2,n-1):
                            kc=logical_k[i,:kl].cpu().float();vc=logical_v[i,:kl].cpu().float()
                            for pos in sorted({0,ql-1}):
                                for head in (0,7,8,15):
                                    scores=(kc[:,head//8]@q[begin+pos,head].cpu().float())*128**-.5
                                    scores[kl-ql+pos+1:]=-float('inf');cpu.append(err(expected[begin+pos,head].cpu(),scores.softmax(0)@vc[:,head//8]))
                        begin+=ql
                    assert all(c['passed'] for c in cpu)
                    checks=[]
                    for f in fixtures:
                        check=err(f['out'],expected);assert check['passed'],(name,f['x'],check)
                        bitwise=bool(torch.equal(f['out'],fixtures[0]['out']));assert bitwise,(name,f['x'],'changed arithmetic')
                        checks.append(dict(x=f['x'],reference=check,control_bitwise=bitwise,full_physical_writer_exact=True,output_padding_preserved=True,segment_scratch_tail_preserved=True,RNG_unchanged=True))
                    phases.append(dict(phase=phase,lengths=list(lengths),query_K_V_changed=bool(phase),lengths_changed=lengths!=cases[case_id][2],slot_mapping_changed=bool(phase and lengths!=cases[case_id][2]),checks=checks,CPU_reference_checks=cpu))
                medians={}
                for trial in range(7):
                    for i in (range(5) if trial%2==0 else reversed(range(5))):
                        f=fixtures[i];start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):f['graph'].replay()
                        end.record();end.synchronize();ms=start.elapsed_time(end)/20
                        data['events'].append(dict(case=case_id,x=f['x'],trial=trial,ms=ms))
                for f in fixtures:
                    vals=[e['ms'] for e in data['events'] if e['case']==case_id and e['x']==f['x']]
                    medians[str(f['x'])]=statistics.median(vals)
                metadata={str(x):modules[x][2].metadata[-1] for x in (0,*CHUNKS)}
                data['cases'].append(dict(case=case_id,name=name,queries=qlens,initial_lengths=cases[case_id][2],strided=strided,phases=phases,medians_ms=medians,actual_attention_metadata=metadata,writer_included_in_events=True))
                data['phase']='cases';save();print(json.dumps(dict(case=name,medians_ms=medians)),flush=True)
                del fixtures,expected;torch.cuda.empty_cache()
        assert len(data['cases'])==8 and len(data['events'])==280
        for name,sha in frozen.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
        for name,sha in data['generated_sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
        data.update(phase='complete',complete=True,all_outputs_control_bitwise=True,changed_Graph_replays_verified=True,full_physical_storage_writer_verified=True)
    except BaseException as error:
        import traceback
        data.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
    finally:signal.alarm(0);save()
    return 0 if data['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
