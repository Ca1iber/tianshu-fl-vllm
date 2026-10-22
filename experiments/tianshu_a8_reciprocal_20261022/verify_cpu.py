# SPDX-License-Identifier: Apache-2.0
"""Independent finite-BF16 domain, complete GPU q/scale, timing verification; no GPU imports."""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def float_words(w):return (np.asarray(w,np.uint16).astype(np.uint32)<<16).view(np.float32)
def reference(words):
 a=float_words(words);peak=np.maximum(abs(a).max(1),np.float32(1e-8));s=(peak.astype(np.float64)/127.).astype(np.float32);assert np.array_equal(s.view(np.uint32),(peak/np.float32(127)).view(np.uint32));division=(a.astype(np.float64)/s[:,None].astype(np.float64)).astype(np.float32);q=np.clip(np.rint(division),-127,127).astype(np.int8);q_direct=np.clip(np.rint(a/s[:,None]),-127,127).astype(np.int8);assert np.array_equal(q,q_direct);return q,s

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);args=p.parse_args();root=args.input.resolve();assert root.is_relative_to('/workspace/benchmark-results');state=json.loads((root/'result.json').read_text());assert state['complete'];assert all(sha(p)==h for p,h in state['sources'].items());domain=state['domain'];assert domain['guards'] and domain['input_unchanged'] and domain['RNG_unchanged']
 positive=np.arange(0x7f80,dtype=np.uint16);ps=positive[float_words(positive)>=np.float32(1e-8)];allwords=np.arange(65536,dtype=np.uint16);values=float_words(allwords);tiny=allwords[np.isfinite(values)&(abs(values)<np.float32(1e-8))];extra=np.zeros(((len(tiny)+4095)//4096,4096),np.uint16);extra.ravel()[:len(tiny)]=tiny
 inp=np.load(root/'domain_A_words.npy',mmap_mode='r',allow_pickle=False);assert inp.dtype==np.uint16 and inp.shape==(len(ps)+len(extra),4096) and domain['M']==len(inp) and domain['K']==4096 and domain['peaks']==len(ps) and domain['clamp_words']==len(tiny);assert sha(root/'domain_A_words.npy')==domain['input_sha256'];assert np.array_equal(inp[len(ps):],extra)
 outputs={}
 for name in ('old','half_div','half_fma'):
  qpath=root/f'domain_{name}_q.npy';spath=root/f'domain_{name}_scale.npy';assert sha(qpath)==domain[name]['q_sha256'] and sha(spath)==domain[name]['scale_sha256'];q=np.load(qpath,mmap_mode='r',allow_pickle=False);s=np.load(spath,mmap_mode='r',allow_pickle=False);assert q.dtype==np.int8 and q.shape==inp.shape and s.dtype==np.float32 and s.shape==(len(inp),);outputs[name]=(q,s)
 checked=0
 for start in range(0,len(inp),128):
  stop=min(start+128,len(inp));words=np.asarray(inp[start:stop]);count=min(stop,len(ps))-start
  if count>0:
   expected=np.zeros((count,4096),np.uint16);peaks=ps[start:start+count];exponent=(peaks>>7).astype(np.int32)
   for d in range(10):
    codes=((exponent[:,None]-d)*128+np.arange(128,dtype=np.int32)).astype(np.uint16);codes=np.where(codes<=peaks[:,None],codes,0).astype(np.uint16);expected[:,d*128:(d+1)*128]=codes;expected[:,1280+d*128:1280+(d+1)*128]=codes|np.uint16(0x8000)
   assert np.array_equal(words[:count],expected)
  ref,sc=reference(words)
  for name,(q,s) in outputs.items():assert np.array_equal(np.asarray(q[start:stop]),ref) and np.array_equal(np.asarray(s[start:stop]).view(np.uint32),sc.view(np.uint32)),(start,name)
  checked+=words.size
 zero_bound=127*(255/128)*2.**-10*(1+5.e-7);assert zero_bound<.5
 cases=state['cases'];expected={(8,2048,False),(64,2048,False),(64,6144,False),(17,259,True),(512,2048,False),(4096,2048,False)};assert len(cases)==6 and {(c['M'],c['K'],c['strided']) for c in cases}==expected;timings=[];perf_values=0
 for c in cases:
  assert c['complete'] and len(c['phases'])==3;m,k=c['M'],c['K'];rng=np.random.default_rng(202610220000+m+k);a=rng.standard_normal((m,k),dtype=np.float32);u=a.view(np.uint32);base=((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16);base[0]=0;peak=12847+30*128;base[1]=0;base[1,:3]=[peak,peak-128,(peak-128)|0x8000]
  for phase,entry in enumerate(c['phases']):
   assert entry['phase']==phase and sha(root/entry['artifact'])==entry['sha256'] and entry['guards'] and entry['input_unchanged'] and entry['RNG_unchanged']
   with np.load(root/entry['artifact'],allow_pickle=False) as f:
    factor=np.float32(1 if phase==0 else -.5 if phase==1 else 2);expected_words=((float_words(base)*factor).view(np.uint32)>>16).astype(np.uint16);assert np.array_equal(f['A_words'],expected_words);ref,sc=reference(f['A_words'])
    for name in ('old','half_div','half_fma'):assert np.array_equal(f[name+'_q'],ref) and np.array_equal(f[name+'_scale'].view(np.uint32),sc.view(np.uint32))
    perf_values+=m*k
  rounds=[r for r in state['rounds'] if (r['M'],r['K'])==(m,k)];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7)) and all(r['nodes_per_graph']==32 and r['replays']==10 and set(r['order'])=={'old','half_div','half_fma'} for r in rounds)
  medians={name:float(np.median([r['milliseconds_per_quantization'][name] for r in rounds])) for name in ('old','half_div','half_fma')};assert all(math.isfinite(v) and v>0 for v in medians.values());timings.append(dict(M=m,K=k,strided=c['strided'],**{name+'_us':value*1000 for name,value in medians.items()},half_div_speedup_pct=(medians['old']/medians['half_div']-1)*100,half_fma_speedup_pct=(medians['old']/medians['half_fma']-1)*100))
 assert all(sha(p)==h for p,h in state['sources'].items());assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(timings[0]),lineterminator='\n');writer.writeheader();writer.writerows(timings)
 result=dict(complete=True,CPU_only=True,all_GPU_q_and_scales_bitwise=True,finite_peak_values=len(ps),clamp_finite_values=len(tiny),domain_elements_per_variant=checked,variants=3,performance_phase_elements_per_variant=perf_values,performance_cases=6,performance_phases=18,unrepresented_relative_exponent_zero_bound=zero_bound,input_guard_RNG_checks=True,original_checkpoint_not_read_or_written=True,model_executed=False,model_quality_not_yet_proven=True,adopted=False,raw_directory=str(root),raw_result_sha256=sha(root/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'))
 (BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
