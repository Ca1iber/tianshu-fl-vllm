# SPDX-License-Identifier: Apache-2.0
"""CPU-only group-size error analysis, no GPU or model execution."""
import csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;path=EXP/'tianshu_native_dotsaddz_down42_20261023/common.py';spec=importlib.util.spec_from_file_location('_W4_group_cpu_reference',path);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
proof=json.loads((EXP/'tianshu_native_dotsaddz_down42_20261023/CPU_GATE_32g.json').read_text());assert proof['complete'] and c.sha(c.MODEL)==proof['checkpoint_sha256'];rows=[]
for layer in (0,20,41):
 wf=c.f32(c.weight(layer));refs={}
 for phase in (0,1,2):
  r=next(v for v in proof['references'] if v['layer']==layer and v['phase']==phase);assert c.sha(r['file'])==r['sha256'];refs[phase]=np.load(r['file'],allow_pickle=False)
 for group in (32,64,128,256):
  v=wf.reshape(2048,6144//group,group);s=c.bf16(np.maximum(abs(v).max(2),np.float32(1e-8))/np.float32(7));q=np.clip(np.rint(v/c.f32(s)[:,:,None]),-8,7).astype(np.int8);deq=c.f32(c.bf16(q.astype(np.float32)*c.f32(s)[:,:,None])).reshape(2048,6144);weight_error=float(np.linalg.norm((deq-wf).astype(np.float64))/np.linalg.norm(wf.astype(np.float64)));signatures=dict(weight_sha256=c.ahash(c.weight(layer)),scale_BF16_sha256=c.ahash(s),signed_q_sha256=c.ahash(q))
  for phase,ref in refs.items():
   projected=c.f32(ref['H']).astype(np.float64)@deq.T.astype(np.float64);original=ref['original'];d=projected-original;error=float(np.linalg.norm(d)/max(np.linalg.norm(original),1e-30));anchor=None
   if group==128:
    difference=abs(projected-ref['quantized']);anchor=float(difference.max());assert anchor<=5e-12*max(float(np.sqrt(np.mean(ref['quantized']**2))),1e-20)
   rows.append(dict(layer=layer,phase=phase,group_size=group,projection_relative_L2=error,maximum_abs_projection_error=float(abs(d).max()),weight_relative_L2=weight_error,previous_G128_FP64_anchor_max_diff=anchor,cells=64*2048,**signatures))
  print(json.dumps(dict(layer=layer,group_size=group,projection_relative_L2=[r['projection_relative_L2'] for r in rows if r['layer']==layer and r['group_size']==group])),flush=True)
with (BASE/'KEY_ERRORS_32g.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
assert c.sha(c.MODEL)==proof['checkpoint_sha256'] and not any(k.split('.')[0] in ('torch','triton','vllm','flag_gems') for k in sys.modules)
g=dict(complete=True,CPU_only=True,no_GPU_imports=True,model_not_executed=True,groups=[32,64,128,256],layers=[0,20,41],phases=3,full_K_projection_cells=len(rows)*64*2048,existing_G128_complete_FP64_reference_anchors_passed=True,source_sha256=c.sha(__file__),parent_helper_sha256=c.sha(path),parent_CPU_gate_sha256=c.sha(EXP/'tianshu_native_dotsaddz_down42_20261023/CPU_GATE_32g.json'),checkpoint_sha256=proof['checkpoint_sha256'],key_errors_sha256=c.sha(BASE/'KEY_ERRORS_32g.csv'),statistics={str(group):dict(mean_projection_relative_L2=float(np.mean([r['projection_relative_L2'] for r in rows if r['group_size']==group])),maximum_projection_relative_L2=max(r['projection_relative_L2'] for r in rows if r['group_size']==group)) for group in (32,64,128,256)},GPU_formats_of_new_groups_not_yet_tested=True,performance_not_tested=True,model_quality_not_tested=True,adopted=False);(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps(g))
