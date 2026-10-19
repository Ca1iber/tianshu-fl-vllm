# SPDX-License-Identifier: Apache-2.0
"""Real layer0 checkpoint matrices: isolated dot dtype and storage experiments."""
import argparse,hashlib,inspect,json,os,signal,statistics,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from plans import SHAPES,MODES,configurations


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);a=p.parse_args()
 root=a.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
 remaining=int((datetime.fromisoformat(a.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>900;signal.alarm(min(remaining,1800))
 before=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in before;(root/'hardware_before.txt').write_text(before)
 import torch,vllm_fl,flag_gems
 from safetensors import safe_open
 import kernel
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 assert getattr(flag_gems,'current_work_registrar',None) is None
 base=Path(__file__).resolve().parent;gate=json.loads((base/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['tile_and_splitK_coverage_cases']==16
 for name,sha in gate['sources'].items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==sha
 paths=[base/x for x in ('kernel.py','plans.py','check_cpu.py','probe.py','run_probe.py','CPU_GATE_32g.json')]
 paths.extend([Path(inspect.getsourcefile(flag_gems.mm)),Path('/workspace/MiniCPM5-2B/config.json')])
 frozen={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
 for name,sha in frozen.items():
  target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
 d=dict(phase='preparing',complete=False,runner_pid=os.getpid(),frozen_sources=frozen,cases=[],events=[],weights={},official_benchmark=False,adopted=False,FP32_accumulation=True,BF16_public_input_output=True,
    scope='Component GEMMs on real layer0 weights with synthetic activations; no model service or accuracy claim',
    static_FP16_weight_conversion_excluded_from_timing=True,checkpoint_files_never_modified=True)
 def save():
  tmp=root/'result.tmp';tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(root/'result.json')
 def check(out,ref,cpu,rows,cols):
  delta=(out.float()-ref).abs();bad=delta>.02+.01*ref.abs()
  sampled=out[rows[:,None],cols[None,:]].cpu().float();cd=(sampled-cpu).abs();cb=cd>.02+.01*cpu.abs()
  return dict(passed=bool(torch.isfinite(out).all() and not bad.any() and not cb.any()),max_abs=float(delta.max()),failed_elements=int(bad.sum()),CPU_max_abs=float(cd.max()),CPU_failed_elements=int(cb.sum()),CPU_expected=cpu.tolist(),CPU_actual=sampled.tolist(),atol=.02,rtol=.01)
 save()
 try:
  files=list(Path('/workspace/MiniCPM5-2B').glob('*.safetensors'));assert len(files)==1
  model_prefix='model.layers.0.'
  keys={'qkv':[model_prefix+'self_attn.'+k+'_proj.weight' for k in ('q','k','v')],
        'o':[model_prefix+'self_attn.o_proj.weight'],
        'gateup':[model_prefix+'mlp.'+k+'_proj.weight' for k in ('gate','up')],
        'down':[model_prefix+'mlp.down_proj.weight']}
  cpu_weights={}
  with safe_open(str(files[0]),framework='pt',device='cpu') as f:
   for name,parts in keys.items():
    tensors=[f.get_tensor(k) for k in parts];w=torch.cat(tensors,0) if len(tensors)>1 else tensors[0].contiguous()
    assert w.dtype==torch.bfloat16;cpu_weights[name]=w
    half=w.to(torch.float16);err=(w.float()-half.float()).abs()
    d['weights'][name]=dict(keys=parts,shape=list(w.shape),dtype=str(w.dtype),BF16_bits_sha256=hashlib.sha256(w.view(torch.int16).numpy().tobytes()).hexdigest(),FP16_conversion_max_abs=float(err.max()),FP16_conversion_changed_elements=int((err!=0).sum()),FP16_conversion_finite=bool(torch.isfinite(half).all()))
  d['phase']='cases';save();case_id=0
  with torch.inference_mode():
   for m in (64,2048):
    for name,n,k in SHAPES:
     assert tuple(cpu_weights[name].shape)==(n,k)
     for c in configurations(m,n):
      torch.manual_seed(2026101950+m+n+k);x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
      w=cpu_weights[name].cuda();w16=w.to(torch.float16);scratch={mode:kernel.prepare(x,w16 if mode==MODES[3] else w,c) for mode in MODES[1:]}
      fs={MODES[0]:lambda:flag_gems.mm(x,w.T),MODES[1]:lambda:kernel.mm(x,w,scratch[MODES[1]],c,False),
          MODES[2]:lambda:kernel.mm(x,w,scratch[MODES[2]],c,True),MODES[3]:lambda:kernel.mm(x,w16,scratch[MODES[3]],c,True)}
      row=dict(case=case_id,projection=name,M=m,N=n,K=k,configuration=c,variants={},phases=[],
          actual_GPU_FP16_storage_conversion_max_abs=float((w.float()-w16.float()).abs().max()),
          actual_GPU_FP16_storage_finite=bool(torch.isfinite(w16).all()),activation_initial_abs_max=float(x.float().abs().max()))
      graphs={};outputs={};actual=None
      for mode in MODES:
       try:
        for _ in range(3):actual=fs[mode]()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):actual=fs[mode]()
        graph.replay();torch.cuda.synchronize();graphs[mode]=graph;outputs[mode]=actual
        row['variants'][mode]=dict(captured=True,compiled_metadata=dict(kernel.LAST) if mode!=MODES[0] else None)
       except Exception as error:
        row['variants'][mode]=dict(captured=False,error=repr(error))
        if any(t in str(error).lower() for t in ('illegal memory','device-side assert')):raise
      rows=torch.linspace(0,m-1,8).long().cuda();cols=torch.linspace(0,n-1,32).long().cuda()
      for phase in (0,1):
       if phase:x.mul_(-.5);w.mul_(2);w16.mul_(2)
       ref=x.float()@w.float().T
       cpu=x[rows].cpu().float()@w[cols].cpu().float().T
       for graph in graphs.values():graph.replay()
       torch.cuda.synchronize();checks={mode:check(out,ref,cpu,rows,cols) for mode,out in outputs.items()}
       for mode,out in outputs.items():
        checks[mode]['exact_to_flag_gems_BF16']=bool(torch.equal(out,outputs[MODES[0]])) if MODES[0] in outputs else None
       row['phases'].append(dict(phase=phase,inputs_and_both_weight_storages_changed=bool(phase),checks=checks))
      passing=[mode for mode in MODES if mode in graphs and all(ph['checks'][mode]['passed'] for ph in row['phases'])]
      for trial in range(7):
       for mode in (passing if trial%2==0 else list(reversed(passing))):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
        for _ in range(20):graphs[mode].replay()
        end.record();end.synchronize();d['events'].append(dict(case=case_id,mode=mode,trial=trial,ms=start.elapsed_time(end)/20))
      row['median_ms']={mode:statistics.median(e['ms'] for e in d['events'] if (e['case'],e['mode'])==(case_id,mode)) for mode in passing}
      row['numerical_passing_modes']=passing;d['cases'].append(row);save()
      print(json.dumps(dict(case=case_id,shape=[m,n,k],configuration=c,median_ms=row['median_ms'],errors={mode:v.get('error') for mode,v in row['variants'].items() if not v['captured']})),flush=True)
      graphs.clear();outputs.clear();fs.clear();scratch.clear();del x,w,w16,ref,cpu,actual;torch.cuda.empty_cache();case_id+=1
  assert len(d['cases'])==16
  assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha for name,sha in frozen.items())
  d.update(phase='complete',complete=True)
 except BaseException as error:
  import traceback
  d.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
 finally:signal.alarm(0);save()
 return 0 if d['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
