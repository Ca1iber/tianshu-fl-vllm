# SPDX-License-Identifier: Apache-2.0
import os,sys,signal,time,runpy,subprocess,traceback
from common import *
def main():
    signal.alarm(900);assert time.time()<1792771080-1200 and not GPU.exists();GPU.mkdir();gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert gate['complete'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items()) and sha(MODEL)==gate['checkpoint_sha256']
    hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(GPU/'hardware_before.txt').write_text(hw)
    state=dict(complete=False,phase='initializing',numeric=[],rounds=[],checkpoint_before=sha(MODEL),adopted=False,model_quality_tested=False,official_benchmark=False)
    def save():(GPU/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        profile=json.loads(PROFILE.read_text());env={k:v.replace(profile['old_output_root'],str(GPU)) for k,v in profile['environment'].items()};os.environ.update(env);sys.path[:0]=env['PYTHONPATH'].split(':');runpy.run_path(str(Path(env['PYTHONPATH'])/'sitecustomize.py'))
        import torch,vllm_fl,flag_gems,kernel
        torch.set_num_threads(4);immutable=[];guards=[]
        def tensor(a,name):
            a=np.ascontiguousarray(a);t=torch.from_numpy(a.copy());t=t.view(torch.bfloat16) if a.dtype==np.uint16 else t;raw=torch.full((a.nbytes+8192,),113,dtype=torch.uint8,device='cuda');v=raw[4096:-4096].view(t.dtype).view(t.shape);v.copy_(t);immutable.append((name,v,ahash(a),int(v._version)));guards.append(raw);return v
        w=tensor(np.load(INPUTS/'WD.npy'),'WD');s=tensor(np.load(INPUTS/'S4.npy'),'S4');z=tensor(np.load(INPUTS/'ZD.npy'),'ZD');xfull=tensor(rows(INPUTS/'X0.npy',64),'X');rng_cpu=torch.get_rng_state().clone();rng_gpu=torch.cuda.get_rng_state().clone()
        def words(t):return t.contiguous().view(torch.uint16).cpu().numpy().copy()
        graphs={};outputs={};state['phase']='capture_all64';save()
        for m in range(1,65):
            x=xfull[:m]
            for _ in range(2):kernel.run(x,w,s,z)
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=kernel.run(x,w,s,z)
            graphs[m]=graph;outputs[m]=out
        for phase in range(3):
            arr=rows(INPUTS/f'X{phase}.npy',64);xfull.copy_(torch.from_numpy(arr).view(torch.bfloat16))
            for i,e in enumerate(immutable):
                if e[0]=='X':immutable[i]=(e[0],e[1],ahash(arr),int(xfull._version));break
            anchors={}
            for b in (32,64):
                entry=next(a for a in gate['anchors'] if a['bucket']==b and a['phase']==phase);anchors[b]=np.roll(np.load(entry['file']),entry['roll'],axis=0)
            state['phase']=f'phase{phase}_all64_numeric';save()
            for m in range(1,65):
                b=bucket(m);x=xfull[:m];raw=torch.full((b*K*2+8192,),113,dtype=torch.uint8,device='cuda');padded=raw[4096:-4096].view(torch.bfloat16).view(b,K);kernel.pad(x,padded);torch.cuda.synchronize();pwords=words(padded)
                assert np.array_equal(pwords[:m],arr[:m]) and (pwords[m:]==0).all() and bool((raw[:4096]==113).all() and (raw[-4096:]==113).all()),(m,phase,'padding')
                expected=anchors[b][:m].copy();expected_hash=ahash(expected);record=dict(M=m,bucket=b,phase=phase,padded_X_full_prefix_tail_guards_exact=True,expected_words_sha256=expected_hash)
                for mode in ('eager','graph'):
                    if mode=='eager':out=kernel.run(x,w,s,z)
                    else:outputs[m].fill_(float('nan'));graphs[m].replay();out=outputs[m]
                    torch.cuda.synchronize();actual=words(out);same=np.array_equal(actual,expected);record[mode+'_words_sha256']=ahash(actual);record[mode+'_all_output_finite']=bool(np.isfinite(f32(actual)).all())
                    if not same:
                        p=GPU/f'failure_M{m}_phase{phase}_{mode}.npy';np.save(p,actual,allow_pickle=False);state['failure_output']=dict(file=str(p),sha256=sha(p));save();raise AssertionError((m,phase,mode,'independent_bucket_prefix_mismatch'))
                state['numeric'].append(record);save()
        arr=rows(INPUTS/'X0.npy',64);xfull.copy_(torch.from_numpy(arr).view(torch.bfloat16))
        for i,e in enumerate(immutable):
            if e[0]=='X':immutable[i]=(e[0],e[1],ahash(arr),int(xfull._version));break
        state['phase']='timing';save()
        for m in TIMING_MS:
            x=xfull[:m]
            direct=lambda:torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,w,128,s,z,size_n=N)
            padded=lambda:kernel.run(x,w,s,z)
            graph_direct=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_direct):out_direct=direct()
            for mode in ('eager','graph'):
                for rid in range(7):
                    ops={'direct':direct,'padded':padded} if mode=='eager' else {'direct':graph_direct.replay,'padded':graphs[m].replay};order=list(ops)
                    if rid%2:order.reverse()
                    times={}
                    for name in order:
                        a,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                        for _ in range(20):ops[name]()
                        e.record();e.synchronize();times[name]=a.elapsed_time(e)/20
                    r=dict(M=m,mode=mode,round=rid,excluded=rid==0,replays=20,order=order,milliseconds=times);state['rounds'].append(r);save();print(json.dumps(r),flush=True)
        state['phase']='checking_immutability';save()
        for name,t,h,version in immutable:assert t._version==version and ahash(t.contiguous().view(torch.uint8).cpu().numpy())==h,name
        for raw in guards:assert bool((raw[:4096]==113).all() and (raw[-4096:]==113).all())
        assert torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_gpu,torch.cuda.get_rng_state()) and sha(MODEL)==gate['checkpoint_sha256'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items())
        state.update(complete=True,phase='complete',checkpoint_sources_full_inputs_versions_guards_RNG_unchanged=True);save()
    except BaseException as error:state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
    finally:(GPU/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
