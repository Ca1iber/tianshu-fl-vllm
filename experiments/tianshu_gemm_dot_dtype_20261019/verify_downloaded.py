# SPDX-License-Identifier: Apache-2.0
"""Audit unchanged GEMM sources, serialized CPU samples and all raw event medians."""
import argparse,hashlib,json,math,statistics,struct
from pathlib import Path
from plans import SHAPES,MODES,configurations
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
d=read(root/'result.json');assert d['phase']=='complete' and d['complete'] and not d['official_benchmark'] and not d['adopted']
assert d['FP32_accumulation'] and d['BF16_public_input_output'] and d['static_FP16_weight_conversion_excluded_from_timing'] and d['checkpoint_files_never_modified']
repo=Path(__file__).resolve().parents[2]
for name,h in d['frozen_sources'].items():
 assert sha(root/'source_snapshot'/Path(name).relative_to('/'))==h
 if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):assert sha(repo/Path(name).relative_to('/workspace/vllm-plugin-FL'))==h
expect=[(m,name,n,k,c) for m in (64,2048) for name,n,k in SHAPES for c in configurations(m,n)]
assert len(d['cases'])==len(expect)==16
sample_values=passing_variants=phase_checks=0
for case,(r,e) in enumerate(zip(d['cases'],expect)):
 m,name,n,k,c=e
 assert (r['case'],r['M'],r['projection'],r['N'],r['K'],r['configuration'])==(case,m,name,n,k,c)
 assert set(r['variants'])==set(MODES) and len(r['phases'])==2
 assert r['actual_GPU_FP16_storage_finite'] and r['actual_GPU_FP16_storage_conversion_max_abs']>=0
 captured={mode for mode,v in r['variants'].items() if v['captured']}
 for mode in captured:
  if mode!=MODES[0]:
   md=r['variants'][mode]['compiled_metadata']
   assert md['FP16_DOT']==(mode in MODES[2:]) and md['A_dtype']==md['output_dtype']=='torch.bfloat16'
   assert md['W_dtype']==('torch.float16' if mode==MODES[3] else 'torch.bfloat16')
   assert (md['num_warps'],md['num_stages'])==(c['warps'],c['stages'])
 for phase,ph in enumerate(r['phases']):
  assert ph['phase']==phase and ph['inputs_and_both_weight_storages_changed']==bool(phase) and set(ph['checks'])==captured
  for mode,check in ph['checks'].items():
   assert check['atol']==.02 and check['rtol']==.01
   actual,expected=check['CPU_actual'],check['CPU_expected'];assert len(actual)==len(expected)==8
   diffs=[];bad=0
   for aa,ee in zip(actual,expected):
    assert len(aa)==len(ee)==32
    for av,ev in zip(aa,ee):
     delta=abs(f32(av-ev));limit=f32(f32(f32(.01)*abs(ev))+f32(.02))
     if not math.isfinite(av) or not math.isfinite(ev):assert not check['passed']
     if delta>limit:bad+=1
     diffs.append(delta);sample_values+=1
   assert bad==check['CPU_failed_elements']
   assert abs(max(diffs)-check['CPU_max_abs'])<1e-10
   if check['passed']:
    assert check['failed_elements']==bad==0 and math.isfinite(check['max_abs'])
   phase_checks+=1
 passing=[mode for mode in MODES if mode in captured and all(ph['checks'][mode]['passed'] for ph in r['phases'])]
 assert r['numerical_passing_modes']==passing and set(r['median_ms'])==set(passing)
 for mode in MODES:
  events=[x for x in d['events'] if (x['case'],x['mode'])==(case,mode)]
  if mode in passing:
   assert len(events)==7 and {x['trial'] for x in events}==set(range(7)) and all(x['ms']>0 for x in events)
   assert statistics.median(x['ms'] for x in events)==r['median_ms'][mode];passing_variants+=1
  else:assert not events
for name,n,k in SHAPES:
 w=d['weights'][name];assert w['shape']==[n,k] and w['dtype']=='torch.bfloat16' and len(w['BF16_bits_sha256'])==64
 assert w['FP16_conversion_finite'] and w['FP16_conversion_max_abs']>=0 and w['FP16_conversion_changed_elements']>=0
assert len(d['events'])==passing_variants*7
summary={}
for m in (64,2048):
 for name,n,k in SHAPES:
  cases=[r for r in d['cases'] if (r['M'],r['projection'])==(m,name)]
  best={mode:min(r['median_ms'][mode] for r in cases if mode in r['median_ms']) for mode in MODES}
  summary[f'{m}_{name}']=best
report=dict(complete=True,frozen_sources=len(d['frozen_sources']),real_layer0_weight_matrices=4,cases=16,passing_variants=passing_variants,changed_phase_checks=phase_checks,independently_recomputed_CPU_sample_values=sample_values,event_records=len(d['events']),timings_ms=summary,
 scope='Recorded full-GPU gate flags plus independently recomputed serialized CPU sample errors and frozen source/event hashes; no full GPU tensors or model accuracy proof',official_benchmark=False,adopted=False)
(root/'local_download_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
