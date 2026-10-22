# SPDX-License-Identifier: Apache-2.0
"""Independent full GPU fixture quantization and timing audit, without GPU imports."""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def f32(words):return (np.asarray(words,np.uint16).astype(np.uint32)<<16).view(np.float32)
def round_bf16(values):
 integers=np.asarray(values,np.float32).view(np.uint32);return ((integers+32767+((integers>>16)&1))>>16).astype(np.uint16)
def expected_input(m,k,phase):
 sample=np.random.default_rng(202610220000+m+k).standard_normal((m,k),dtype=np.float32);original=round_bf16(sample)
 original[0]=0;maximum=12847+30*128;original[1]=0;original[1,:3]=[maximum,maximum-128,(maximum-128)|32768]
 original[2,:7]=round_bf16(np.array([0,.5,1.5,2.5,-.5,-1.5,-2.5],np.float32));original[3]=1
 actual=f32(original)
 if phase==1:actual=np.multiply(actual,np.float32(-.5))
 elif phase==2:actual=actual[:,::-1].copy()+np.float32(.125)
 result=round_bf16(actual);result[0]=0;return result

def reference(words):
 values=f32(words);peak=np.maximum(np.max(np.abs(values),axis=1),np.float32(1e-8));scales=np.asarray(peak.astype(np.float64)/127.,np.float32)
 assert np.array_equal(scales.view(np.uint32),(peak/np.float32(127.)).view(np.uint32))
 divided=np.asarray(values.astype(np.float64)/scales[:,None].astype(np.float64),np.float32);codes=np.clip(np.rint(divided),-127,127).astype(np.int8)
 assert np.array_equal(codes,np.clip(np.rint(values/scales[:,None]),-127,127).astype(np.int8));return codes,scales

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args();root=args.input.resolve();assert root.is_relative_to('/workspace/benchmark-results')
 state=json.loads((root/'result.json').read_text());assert state['complete'] and state['model_executed'] is False and state['adopted'] is False
 assert all(sha(path)==digest for path,digest in state['sources'].items())
 configs=json.loads((BASE/'configs.json').read_text());variants=['old']+[f'R{r}W{w}' for r,w in configs]
 expected={(8,2048,False),(64,2048,False),(64,6144,False),(17,259,True),(128,2048,False),(512,2048,False),(2048,2048,False),(4096,2048,False)}
 assert len(state['cases'])==8 and {(c['M'],c['K'],c['strided']) for c in state['cases']}==expected
 timings=[];elements=0;phases=0
 for case in state['cases']:
  m,k=case['M'],case['K'];assert case['complete'] and len(case['phases'])==3 and set(case['compiled'])==set(variants)
  previous_words=None
  for phase,entry in enumerate(case['phases']):
   assert entry['phase']==phase and all(entry[key] for key in ('all_variants_q_scale_equal','guards','RNG_unchanged','input_unchanged')) and sha(root/entry['artifact'])==entry['sha256']
   with np.load(root/entry['artifact'],allow_pickle=False) as artifact:
    words=artifact['A_words'];assert words.dtype==np.uint16 and words.shape==(m,k) and np.array_equal(words,expected_input(m,k,phase))
    if previous_words is not None:assert not np.array_equal(previous_words,words)
    previous_words=words.copy();codes,scales=reference(words)
    assert set(artifact.files)=={'A_words'}|{v+suffix for v in variants for suffix in ('_q','_scale')}
    for variant in variants:
     q,s=artifact[variant+'_q'],artifact[variant+'_scale'];assert q.dtype==np.int8 and q.shape==(m,k) and s.dtype==np.float32 and s.shape==(m,)
     assert np.array_equal(q,codes) and np.array_equal(s.view(np.uint32),scales.view(np.uint32)),(m,k,phase,variant)
    elements+=m*k;phases+=1
  rounds=[entry for entry in state['rounds'] if (entry['M'],entry['K'])==(m,k)]
  assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7))
  assert all(r['nodes_per_graph']==32 and r['replays']==10 and set(r['order'])==set(variants) and set(r['milliseconds_per_quantization'])==set(variants) for r in rounds)
  medians={v:float(np.median([r['milliseconds_per_quantization'][v] for r in rounds])) for v in variants}
  assert all(math.isfinite(value) and value>0 for value in medians.values())
  for rows,warps in configs:
   variant=f'R{rows}W{warps}';metadata=case['compiled'][variant];old,candidate=medians['old'],medians[variant]
   timings.append(dict(M=m,K=k,strided=case['strided'],rows_per_CTA=rows,warps=warps,old_us=old*1000,candidate_us=candidate*1000,speedup_pct=(old/candidate-1)*100,n_regs=metadata['n_regs'],n_spills=metadata['n_spills'],shared=metadata['shared']))
 assert len(timings)==112 and len(state['rounds'])==56 and phases==24
 assert all(sha(path)==digest for path,digest in state['sources'].items())
 assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(timings[0]),lineterminator='\n');writer.writeheader();writer.writerows(timings)
 gate=dict(complete=True,CPU_only=True,all_saved_GPU_q_and_scales_bitwise=True,configurations=14,performance_cases=8,performance_phases=phases,phase_elements_per_variant=elements,variants_including_old=len(variants),timing_rows=len(timings),input_guard_RNG_checks=True,production_NT_NN_AST_equal=True,model_executed=False,model_quality_not_yet_proven=True,adopted=False,raw_directory=str(root),raw_result_sha256=sha(root/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'))
 (BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
