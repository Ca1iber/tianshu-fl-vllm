# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,json,os,signal,subprocess,traceback
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir();signal.alarm(120)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
sources={str(path):sha(path) for path in [Path(__file__),BASE/'PLAN_32g.md']}
(root/'SOURCES.json').write_text(json.dumps(sources,indent=2)+'\n');hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(root/'hardware_before.txt').write_text(hw)
state=dict(complete=False,sources=sources,cases=[],runner_pid=os.getpid(),model_executed=False,adopted=False)
try:
 import torch,vllm_fl
 state['torch_version']=torch.__version__;state['device']=torch.cuda.get_device_name()
 for m,k in [(64,2048),(64,6144)]:
  n=256;av=((np.arange(m*k,dtype=np.int32).reshape(m,k)*7+3)%31-15).astype(np.int8);wv=((np.arange(k*n,dtype=np.int32).reshape(k,n)*11+5)%29-14).astype(np.int8)
  for layout in ['NN','NT']:
   a=torch.from_numpy(av.copy()).cuda();backing=torch.from_numpy(wv.copy() if layout=='NN' else wv.T.copy()).cuda();w=backing if layout=='NN' else backing.T
   storage=torch.full((m*n+512,),1234567,dtype=torch.int32,device='cuda');out=storage[256:256+m*n].view(m,n)
   entry=dict(M=m,K=k,N=n,layout=layout,weight_stride=list(w.stride()),phases=[]);state['cases'].append(entry)
   torch._int_mm(a,w,out=out);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):torch._int_mm(a,w,out=out)
   for phase in range(3):
    aa=av.copy() if phase==0 else (-av if phase==1 else av[:,::-1].copy());ww=wv.copy() if phase!=2 else -wv
    a.copy_(torch.from_numpy(aa));w.copy_(torch.from_numpy(ww));cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();graph.replay();torch.cuda.synchronize();actual=out.cpu().numpy();expected=aa.astype(np.int64)@ww.astype(np.int64)
    assert np.array_equal(actual.astype(np.int64),expected) and bool((storage[:256]==1234567).all() and (storage[-256:]==1234567).all())
    assert np.array_equal(a.cpu().numpy(),aa) and np.array_equal(w.cpu().numpy(),ww) and torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
    artifact=root/f'M{m}_K{k}_{layout}_phase{phase}.npz';np.savez(artifact,A=aa,W=ww,output=actual)
    entry['phases'].append(dict(phase=phase,full_CPU_INT64_exact=True,guards_inputs_RNG=True,artifact=artifact.name,sha256=sha(artifact)))
   entry['complete']=True;print(json.dumps({key:entry[key] for key in ['M','K','layout','complete']}),flush=True)
 assert all(sha(path)==value for path,value in sources.items());state['complete']=True
except BaseException:state['error']=traceback.format_exc();print(state['error'],flush=True)
finally:
 (root/'result.json').write_text(json.dumps(state,indent=2)+'\n');(BASE/'GPU_SUPPORT_GATE_32g.json').write_text(json.dumps(state,indent=2)+'\n')
raise SystemExit(0 if state['complete'] else 1)
