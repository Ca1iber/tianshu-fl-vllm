import hashlib,json,os
from pathlib import Path
import numpy as np
from domain_cpu import peaks,values_for_peaks,clamp_words,f32,old
BASE=Path(__file__).resolve().parent
ps=peaks();counts=dict(peak_values=len(ps),enumerated_peak_value_pairs=0,simple_reciprocal_integer_mismatches=0,half_div_corrected_mismatches=0,half_fma_formula_mismatches=0,half_decision_mismatches=0,clamp_finite_values=len(clamp_words()))
examples=[]
for start in range(0,len(ps),128):
 words=values_for_peaks(ps[start:start+128]);a=f32(words);reference,scale=old(a);peak=f32(ps[start:start+128]);inv=np.float32(1.)/scale;raw=np.clip(np.rint(np.float32(a*inv[:,None])),-127,127).astype(np.int8);tie=np.abs(a)==peak[:,None]*np.float32(.5);half=np.rint(np.float32(peak*np.float32(.5)/scale));signed=np.where(a<0,-half[:,None],half[:,None]);corrected=np.where(tie,signed,raw).astype(np.int8)
 residual=np.float32(peak.astype(np.float64)*.5-scale.astype(np.float64)*63.5);halffma=np.where(residual< -np.float32(scale*np.float32(2.**-19)),np.float32(63),np.float32(64));fmafix=np.where(tie,np.where(a<0,-halffma[:,None],halffma[:,None]),raw).astype(np.int8)
 counts['enumerated_peak_value_pairs']+=a.size;counts['simple_reciprocal_integer_mismatches']+=int(np.count_nonzero(raw!=reference));counts['half_div_corrected_mismatches']+=int(np.count_nonzero(corrected!=reference));counts['half_fma_formula_mismatches']+=int(np.count_nonzero(fmafix!=reference));counts['half_decision_mismatches']+=int(np.count_nonzero(half!=halffma))
 if len(examples)<8:
  for r,c in np.argwhere(raw!=reference)[:8-len(examples)]:examples.append(dict(peak_word=int(ps[start+r]),a_word=int(words[r,c]),old=int(reference[r,c]),reciprocal=int(raw[r,c]),corrected=int(corrected[r,c])))
assert counts['half_div_corrected_mismatches']==counts['half_fma_formula_mismatches']==counts['half_decision_mismatches']==0
result=dict(complete=True,CPU_only=True,GPU_executed=False,counts=counts,examples=examples,limit='CPU residual computed in FP64 then rounded; actual GPU fma must still be independently checked. No model quality or speed claim.',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),BASE/'domain_cpu.py',BASE/'PLAN_CPU_32g.md']})
(BASE/'CPU_FORMULA_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
