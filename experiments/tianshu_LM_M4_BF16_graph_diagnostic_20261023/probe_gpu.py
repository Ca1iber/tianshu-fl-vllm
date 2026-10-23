# SPDX-License-Identifier: Apache-2.0
import os,sys,json,runpy,signal,subprocess,hashlib,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;SOURCE=EXP/'tianshu_native_W4_lm_small_20261023';sys.path.insert(0,str(SOURCE))
from common import sha,ahash,f32,rows,INPUTS,MODEL,PROFILE,weight
ROOT=Path('/workspace/benchmark-results/20261023_LM_M4_BF16_graph_diagnostic_GPU_tianshu')
def main():
    signal.alarm(300);assert time.time()<1792771080-600 and not ROOT.exists();ROOT.mkdir()
    gate=json.loads((BASE/'CPU_GATE_32g.json').read_text());assert all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items()) and sha(MODEL)==gate['checkpoint_sha256']
    hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(ROOT/'hardware_before.txt').write_text(hw)
    state=dict(complete=False,phase='initializing',pid=os.getpid(),samples=[],checkpoint_before=sha(MODEL),adopted=False,model_quality_tested=False,official_benchmark=False)
    def save():(ROOT/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        profile=json.loads(PROFILE.read_text());env={k:v.replace(profile['old_output_root'],str(ROOT)) for k,v in profile['environment'].items()};os.environ.update(env);sys.path[:0]=env['PYTHONPATH'].split(':');runpy.run_path(str(Path(env['PYTHONPATH'])/'sitecustomize.py'))
        import torch,vllm_fl,flag_gems
        from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
        from vllm_fl.utils import get_flag_gems_whitelist_blacklist
        torch.set_num_threads(4);white,black=get_flag_gems_whitelist_blacklist();assert not white and not black;flag_gems.enable(record=False,once=True);method=UnquantizedEmbeddingMethod();immutable=[];guards=[]
        def tensor(a,name):
            a=np.ascontiguousarray(a);t=torch.from_numpy(a.copy()).view(torch.bfloat16);raw=torch.full((a.nbytes+8192,),113,dtype=torch.uint8,device='cuda');v=raw[4096:-4096].view(t.dtype).view(t.shape);v.copy_(t);immutable.append((name,v,ahash(a),int(v._version)));guards.append(raw);return v
        w=tensor(np.asarray(weight()),'weight');xfull=tensor(rows(INPUTS/'X0.npy',64),'X');x=xfull[:4];layer=SimpleNamespace(weight=w)
        def run():return method.apply(layer,x,None)
        for _ in range(2):run()
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):output=run()
        rng_cpu=torch.get_rng_state().clone();rng_gpu=torch.cuda.get_rng_state().clone();state['phase']='sampling';save()
        def sample(t,phase,mode,iteration):
            torch.cuda.synchronize();a=t.detach().contiguous().view(torch.uint16).cpu().numpy().copy();path=ROOT/f'phase{phase}_{mode}_{iteration}.npy';np.save(path,a,allow_pickle=False);state['samples'].append(dict(phase=phase,mode=mode,iteration=iteration,file=str(path),sha256=sha(path)));save()
        for phase in range(3):
            arr=rows(INPUTS/f'X{phase}.npy',64);xfull.copy_(torch.from_numpy(arr).view(torch.bfloat16))
            for i,e in enumerate(immutable):
                if e[0]=='X':immutable[i]=(e[0],e[1],ahash(arr),int(xfull._version));break
            for iteration in range(3):sample(run(),phase,'eager',iteration)
            for iteration in range(3):graph.replay();sample(output,phase,'graph_unpoisoned',iteration)
            for iteration in range(3):output.fill_(float('nan'));graph.replay();sample(output,phase,'graph_poisoned',iteration)
        for name,t,h,version in immutable:assert t._version==version and ahash(t.contiguous().view(torch.uint8).cpu().numpy())==h,name
        for raw in guards:assert bool((raw[:4096]==113).all() and (raw[-4096:]==113).all())
        assert torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_gpu,torch.cuda.get_rng_state());assert sha(MODEL)==gate['checkpoint_sha256'] and all(sha(p)==h for p,h in gate['sources'].items()) and all(sha(p)==h for p,h in gate['files'].items())
        state.update(complete=True,phase='complete',checkpoint_sources_full_inputs_guards_versions_RNG_unchanged=True);save()
    except BaseException as error:state.update(phase='failed',error=repr(error),traceback=traceback.format_exc());save();raise
    finally:(ROOT/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if __name__=='__main__':main()
