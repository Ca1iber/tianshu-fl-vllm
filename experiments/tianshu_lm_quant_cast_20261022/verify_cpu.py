# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,statistics,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;ROOT=Path('/workspace/benchmark-results/20261022_lm_quant_cast_GPU_tianshu');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bf(words):return (np.asarray(words,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def main():
 import torch
 from safetensors import safe_open
 assert not torch.cuda.is_initialized() and not (BASE/'INDEPENDENT_CPU_GATE_32g.json').exists();s=json.loads((ROOT/'result.json').read_text());assert s['complete'] and s['phase']=='complete' and len(s['phases'])==3 and len(s['rounds'])==14
 for p,h in s['frozen_sources'].items():assert sha(p)==h and sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))==h
 with safe_open(str(MODEL),framework='pt',device='cpu') as cp:weight=cp.get_tensor('lm_head.weight').contiguous();words=weight.view(torch.uint16).numpy().copy()
 wf=bf(words);sc=np.maximum(np.max(np.abs(wf),axis=1),np.float32(1.e-8))/np.float32(127.);codes=np.clip(np.rint(wf/sc[:,None]),-127,127).astype(np.int8);del wf
 assert hashlib.sha256(words.tobytes()).hexdigest()==s['weight_signatures']['original_BF16_sha256'] and hashlib.sha256(codes.tobytes()).hexdigest()==s['weight_signatures']['packed_sha256'] and hashlib.sha256(sc.tobytes()).hexdigest()==s['weight_signatures']['scales_sha256'];cells=0
 for phase in s['phases']:
  art=ROOT/phase['artifact'];assert sha(art)==phase['sha256'];data=np.load(art);sample=np.random.default_rng(phase['seed']).standard_normal((64,2048),dtype=np.float32);bits=sample.view(np.uint32);aw=((bits+32767+((bits>>16)&1))>>16).astype(np.uint16);aw[0]=0;a=bf(aw);asc=np.maximum(np.max(np.abs(a),axis=1),np.float32(1.e-8))/np.float32(127.);q=np.clip(np.rint(a/asc[:,None]),-127,127).astype(np.int8);assert np.array_equal(q,data['Aq']) and np.array_equal(asc,data['As']);r,c=data['rows'],data['columns'];integer=q[r].astype(np.int64)@(codes[c].astype(np.int64)*phase['weight_sign']).T;assert np.array_equal(integer,data['integer_selected']);value=(integer.astype(np.float32)*asc[r,None])*(sc[c][None,:]*np.float32(phase['weight_scale_factor']));vb=value.view(np.uint32);rounded=((vb+32767+((vb>>16)&1))>>16).astype(np.uint16);assert np.array_equal(bf(rounded),data['output_FP32_selected']);cells+=integer.size;assert len(phase['entries'])==4 and all(e['full_FP32_bit_exact'] and e['graph_actual_return_poison_matches_eager'] for e in phase['entries'])
 rows=[];summary={}
 for mode in ('graph','eager'):
  selected=[r for r in s['rounds'] if r['mode']==mode];assert len(selected)==7 and [r['round'] for r in selected]==list(range(7));summary[mode]={}
  for name in selected[0]['milliseconds']:
   median=statistics.median(r['milliseconds'][name] for r in selected[1:]);summary[mode][name]=median
  for r in selected:
   for name,value in r['milliseconds'].items():rows.append(dict(mode=mode,round=r['round'],excluded=r['excluded'],name=name,milliseconds=value))
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 gains={mode:{name:100*(v['control']/value-1) for name,value in v.items() if name!='control'} for mode,v in summary.items()};assert sha(MODEL)==s['checkpoint_sha256_before']==s['checkpoint_sha256_after']=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c';assert not torch.cuda.is_initialized()
 gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,CPU_INT64_and_BF16_FP32_cells=cells,full_GPU_FP32_cells_per_entry=64*130560,phases=3,actual_graph_return_poison_validated=True,original_checkpoint_and_all_weight_pack_signatures_unchanged=True,all_sources_and_snapshots_verified=True,median_ms=summary,speedup_percent=gains,component_screen_gt3pct=any(v>3 for v in gains['graph'].values()),model_executed=False,model_quality_not_tested=True,official_benchmark=False,adopted=False,source_count=len(s['frozen_sources']),raw_result_sha256=sha(ROOT/'result.json'),verifier_sha256=sha(__file__));(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
