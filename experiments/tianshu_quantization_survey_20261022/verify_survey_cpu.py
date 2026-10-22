# SPDX-License-Identifier: Apache-2.0
"""Independent coverage, aggregation and scalar weight-code audit; CPU only."""
import csv,hashlib,json,math,struct
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 gate=json.loads((BASE/'CPU_SURVEY_GATE_32g.json').read_text());assert gate['complete'] and gate['checkpoint_unchanged']
 assert sha(MODEL)==gate['checkpoint_sha256_before']==gate['checkpoint_sha256_after']
 assert all(sha(p)==v for p,v in gate['sources'].items())
 assert sha(BASE/'WEIGHT_ERROR_32g.csv')==gate['weight_error_CSV_sha256'] and sha(BASE/'ROLE_SUMMARY_32g.csv')==gate['role_summary_CSV_sha256']
 with MODEL.open('rb') as f: length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length))
 expected={f'model.layers.{i}.{p}.weight' for i in range(42) for p in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']}|{'lm_head.weight'}
 rows=list(csv.DictReader((BASE/'WEIGHT_ERROR_32g.csv').open()));summaries=list(csv.DictReader((BASE/'ROLE_SUMMARY_32g.csv').open()));formats={'W8_channel':(8,0,False),'W4_sym64':(4,64,False),'W4_sym128':(4,128,False),'W4_aff64':(4,64,True),'W4_aff128':(4,128,True)}
 assert len(rows)==1475 and {(r['tensor'],r['format']) for r in rows}=={(t,f) for t in expected for f in formats}
 assert len(summaries)==40
 for s in summaries:
  r=[r for r in rows if r['role']==s['role'] and r['format']==s['format']];assert len(r)==int(s['tensors'])
  for field in ('elements','error_square_sum','weight_square_sum','payload_bytes','original_bytes'):assert math.isclose(sum(float(x[field]) for x in r),float(s[field]),rel_tol=1.e-12)
  assert math.isclose(math.sqrt(float(s['error_square_sum'])/float(s['weight_square_sum'])),float(s['NRMSE']),rel_tol=1.e-12)
 checks=0;selected=[]
 for name in sorted(expected):
  d=header[name];n,k=d['shape'];words=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+d['data_offsets'][0],shape=(n,k))
  for rowid in {0,n//2,n-1}:
   w=(np.asarray(words[rowid]).astype(np.uint32)<<16).view(np.float32)
   for label,(bits,group,aff) in formats.items():
    g=group or k;ng=(k+g-1)//g
    for j in {0,ng//2,ng-1}:
     values=w[j*g:min((j+1)*g,k)];limit=(1<<bits)-1 if aff else (1<<(bits-1))-1
     if aff:lo=min(0.,float(values.min()));hi=max(0.,float(values.max()));scale=np.float32(max(hi-lo,1.e-8))/np.float32(limit);z=max(0,min(limit,round(float(np.float32(-lo)/scale))))
     else:scale=np.float32(max(float(np.abs(values).max()),1.e-8))/np.float32(limit);z=0
     for idx in {0,len(values)//2,len(values)-1}:
      v=values[idx];q=max(0 if aff else -limit,min(limit,round(float(np.float32(v/scale)))+z));dequant=np.float32(q-z)*scale
      u=struct.unpack('<I',struct.pack('<f',dequant))[0];rounded=((u+0x7fff+((u>>16)&1))>>16)&0xffff;value=struct.unpack('<f',struct.pack('<I',rounded<<16))[0]
      assert math.isfinite(value);checks+=1
  del words
 aggregates={label:{'original_bytes':sum(int(r['original_bytes']) for r in rows if r['format']==label),'payload_bytes':sum(int(r['payload_bytes']) for r in rows if r['format']==label)} for label in formats}
 result=dict(complete=True,CPU_only=True,coverage_tensors=295,coverage_formats=5,aggregate_rows=40,scalar_samples=checks,checkpoint_unchanged=True,all_sources_unchanged=True,aggregates=aggregates,independent_audit_limit='Scalar scan checks finite reconstruction and independent statistics/coverage; full tensor error was computed by the original read-only scanner, not independently recomputed.',model_quality_proven=False,adopted=False)
 (BASE/'CPU_SURVEY_INDEPENDENT_AUDIT_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
