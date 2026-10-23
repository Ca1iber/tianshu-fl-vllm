# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json,signal,statistics,subprocess,time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;OLD=EXP/'tianshu_native_W4_gateup_component_v2_20261023';ROOT=Path('/dev/shm/tianshu_W4_gateup_decomposition_20261023')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 signal.alarm(600);assert time.time()<1792771080-600 and not ROOT.exists();ROOT.mkdir();g=json.loads((OLD/'CPU_GATE_32g.json').read_text());proof=json.loads((OLD/'INDEPENDENT_CPU_GATE_32g.json').read_text());assert proof['numeric_qualified'] and all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items());assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
 import torch,flag_gems
 from flag_gems.fused.silu_and_mul import silu_and_mul_out
 torch.set_num_threads(4);state=dict(complete=False,rounds=[],sources={str(p):sha(p) for p in [BASE/'probe_gpu.py',BASE/'PLAN_32g.md',BASE/'EXECUTION.sha256',OLD/'CPU_GATE_32g.json',OLD/'INDEPENDENT_CPU_GATE_32g.json']},paired_component_only=True)
 def save():(ROOT/'result.json').write_text(json.dumps(state,indent=2)+'\n')
 def tensor(a):
  t=torch.from_numpy(a.copy());return (t.view(torch.bfloat16) if a.dtype==np.uint16 else t).to('cuda')
 for bank in g['banks']:
  arrays=dict(np.load(bank['file']));w,s,z=(tensor(arrays[k]) for k in ('W4','S4','Z4'))
  with np.load(Path(g['root'])/'X0.npz') as f:x=tensor(f['X'])
  y=torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,w,128,s,z,size_n=12288);out=torch.empty((64,6144),device='cuda',dtype=torch.bfloat16)
  methods={'native_W4_mm':lambda:torch.ops.aten._weight_int4pack_mm_dotsaddz.default(x,w,128,s,z,size_n=12288),'same_FlagGems_SiLU':lambda:silu_and_mul_out(y[:,:6144],y[:,6144:],out)};graphs={};alive=[];pool=torch.cuda.graph_pool_handle()
  for name,fn in methods.items():
   for _ in range(3):fn()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,pool=pool):value=fn()
   graphs[name]=graph;alive.append(value)
  for mode in ('eager','graph'):
   for rid in range(7):
    order=list(methods);order=order[::-1] if rid%2 else order;times={}
    for name in order:
     fn=graphs[name].replay if mode=='graph' else methods[name];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
     for _ in range(50):fn()
     b.record();b.synchronize();times[name]=a.elapsed_time(b)/50
    row=dict(layer=bank['layer'],mode=mode,round=rid,excluded=rid==0,ms=times);state['rounds'].append(row);save()
  del graphs,alive,methods
 state.update(complete=True,baseline_saved_combined_graph_ms={str(r['layer']):r['control_ms'] for r in proof['timings'] if r['mode']=='graph'},no_model_or_new_precision_qualification=True);save();rows=[]
 for layer in (0,20,41):
  for mode in ('eager','graph'):
   subset=[r for r in state['rounds'] if r['layer']==layer and r['mode']==mode and not r['excluded']];rows.append(dict(layer=layer,mode=mode,**{k:statistics.median(r['ms'][k] for r in subset) for k in subset[0]['ms']}))
 state['summary']=rows;save();(BASE/'KEY_COMPONENTS_32g.json').write_text(json.dumps(dict(complete=True,rows=rows,raw_root=str(ROOT),raw_result_sha256=sha(ROOT/'result.json')),indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
