# SPDX-License-Identifier: Apache-2.0
import os,sys,signal,time,subprocess,traceback,runpy,inspect
from types import SimpleNamespace
from common import *

def main():
    signal.alarm(900);assert time.time()<1792771080-1200 and not GPU.exists();GPU.mkdir()
    gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items()) and sha(MODEL)==gate['checkpoint_sha256']
    hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(GPU/'hardware_before.txt').write_text(hw)
    state=dict(complete=False,phase='initializing',pid=os.getpid(),numeric=[],rounds=[],routes=[],CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),checkpoint_sha256_before=sha(MODEL),model_quality_tested=False,official_benchmark=False,adopted=False,LM_small_actual_r3_BF16_control=True)
    def save():(GPU/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        profile=json.loads(PROFILE.read_text());env={k:v.replace(profile['old_output_root'],str(GPU)) for k,v in profile['environment'].items()};os.environ.update(env);sys.path[:0]=env['PYTHONPATH'].split(':');runpy.run_path(str(Path(env['PYTHONPATH'])/'sitecustomize.py'))
        import torch,vllm_fl,flag_gems
        from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
        from vllm_fl.utils import get_flag_gems_whitelist_blacklist
        torch.set_num_threads(4);white,black=get_flag_gems_whitelist_blacklist();assert not white and not black;flag_gems.enable(record=False,once=True)
        method=UnquantizedEmbeddingMethod();immutable=[];guards=[]
        def tensor(a,name):
            a=np.ascontiguousarray(a);t=torch.from_numpy(a.copy());t=t.view(torch.bfloat16) if a.dtype==np.uint16 else t
            raw=torch.full((a.nbytes+8192,),113,dtype=torch.uint8,device='cuda');view=raw[4096:-4096].view(t.dtype).view(t.shape);view.copy_(t);guards.append(raw);immutable.append((name,view,ahash(a),int(view._version)));return view
        original=tensor(np.asarray(weight()),'original_weight');packed=tensor(np.load(INPUTS/'WD.npy'),'WD');scales=tensor(np.load(INPUTS/'S4.npy'),'S4');zeros=tensor(np.load(INPUTS/'ZD.npy'),'ZD');xfull=tensor(rows(INPUTS/'X0.npy',64),'X');layer=SimpleNamespace(weight=original)
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
        def set_phase(phase):
            arr=rows(INPUTS/f'X{phase}.npy',64);xfull.copy_(torch.from_numpy(arr).view(torch.bfloat16))
            for i,e in enumerate(immutable):
                if e[0]=='X':immutable[i]=(e[0],e[1],ahash(arr),int(xfull._version));break
        for m in MS:
            x=xfull[:m];operations={VARIANTS[0]:lambda:method.apply(layer,x,None),VARIANTS[1]:lambda:torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,packed,128,scales,zeros,size_n=N)};graphs={};outputs={}
            state['phase']=f'M{m}_warming';save();route=set()
            def trace(frame,event,arg):
                if event=='call' and ('flag_gems' in frame.f_code.co_filename or 'vllm/model_executor/layers' in frame.f_code.co_filename):route.add((frame.f_code.co_filename,frame.f_code.co_name))
            sys.setprofile(trace)
            try:operations[VARIANTS[0]]()
            finally:sys.setprofile(None)
            vendor='/usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar/ops/linear.py'
            assert any(p==vendor and name=='linear' for p,name in route),(m,sorted(route));assert sha(vendor)==gate['sources'][vendor]
            state['routes'].append(dict(M=m,actual_embedding_apply=True,vendor_linear_entered=True,frames=[dict(file=p,function=n,sha256=sha(p)) for p,n in sorted(route) if Path(p).is_file()]));save()
            for name,fn in operations.items():
                for _ in range(2):fn()
                torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):out=fn()
                graphs[name]=g;outputs[name]=out
            state['phase']=f'M{m}_numeric';save()
            for phase in range(3):
                set_phase(phase)
                for name,fn in operations.items():
                    eager=fn();torch.cuda.synchronize();expected=words(eager);outputs[name].fill_(float('nan'));graphs[name].replay();torch.cuda.synchronize();actual=words(outputs[name]);assert np.array_equal(actual,expected) and np.isfinite(f32(actual)).all(),(m,phase,name)
                    path=GPU/f'M{m}_{name}_phase{phase}.npy';np.save(path,actual,allow_pickle=False);state['numeric'].append(dict(M=m,phase=phase,variant=name,file=str(path),sha256=sha(path),full_graph_poison_eager_exact=True,all_output_finite=True));save()
            set_phase(0);state['phase']=f'M{m}_timing';save()
            for mode in ('eager','graph'):
                for rid in range(7):
                    order=list(operations)
                    if rid%2:order.reverse()
                    times={}
                    for name in order:
                        fn=graphs[name].replay if mode=='graph' else operations[name];start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
                        for _ in range(20):fn()
                        end.record();end.synchronize();times[name]=start.elapsed_time(end)/20
                    r=dict(M=m,mode=mode,round=rid,excluded=rid==0,replays=20,order=order,milliseconds=times);state['rounds'].append(r);save();print(json.dumps(r),flush=True)
        state['phase']='checking_immutability';save()
        for name,t,h,version in immutable:assert t._version==version and ahash(t.contiguous().view(torch.uint8).cpu().numpy())==h,name
        for raw in guards:assert bool((raw[:4096]==113).all() and (raw[-4096:]==113).all())
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        state['checkpoint_sha256_after']=sha(MODEL);assert state['checkpoint_sha256_after']==state['checkpoint_sha256_before'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items())
        state.update(complete=True,phase='complete',checkpoint_sources_and_inputs_unchanged=True,all_guards_versions_full_bytes_preserved=True,RNG_unchanged=True);save()
    except BaseException as error:
        state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
    finally:(GPU/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
