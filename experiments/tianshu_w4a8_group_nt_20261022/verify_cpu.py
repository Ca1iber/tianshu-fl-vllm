# SPDX-License-Identifier: Apache-2.0
"""No GPU imports: independent checkpoint reconstruction, FP64 dots, timings audit."""
import argparse,csv,hashlib,json,math,struct,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def words_to_float(w):return (np.asarray(w,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def round_words(x):
 u=np.asarray(x,dtype=np.float32).view(np.uint32);return ((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)
def independently_quantize(w,label):
 n,k=w.shape;bits=8 if label=='W8_channel' else 4;g=k if bits==8 else 128;ng=(k+g-1)//g;aff=label=='W4_aff128'
 padded=np.pad(w,((0,0),(0,ng*g-k))).reshape(n,ng,g)
 if aff:
  low=np.minimum(padded.min(2),np.float32(0));high=np.maximum(padded.max(2),np.float32(0));sc=np.maximum(high-low,np.float32(1.e-8))/np.float32(15);zero=np.clip(np.rint(-low/sc),0,15).astype(np.int16);code=np.clip(np.rint(padded/sc[:,:,None])+zero[:,:,None],0,15)
 else:
  limit=127 if bits==8 else 7;sc=np.maximum(abs(padded).max(2),np.float32(1.e-8))/np.float32(limit);zero=np.zeros((n,ng),np.int16);code=np.clip(np.rint(padded/sc[:,:,None]),-limit,limit)
 reconstructed=((code.astype(np.float32)-zero[:,:,None])*sc[:,:,None]).reshape(n,-1)[:,:k]
 return (code-zero[:,:,None]).reshape(n,-1)[:,:k].astype(np.int8),sc,words_to_float(round_words(reconstructed))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);args=p.parse_args();root=args.input.resolve();assert root.is_relative_to('/workspace/benchmark-results');state=json.loads((root/'result.json').read_text());assert state['complete'] and state['checkpoint_unchanged']
 assert sha(MODEL)==state['checkpoint_sha256_before']==state['checkpoint_sha256_after'];assert all(sha(p)==v for p,v in state['frozen_sources'].items())
 with MODEL.open('rb') as f:length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length))
 decoded_selected={};original_selected={};base8={};decode_count=0;quantized_selected={};group_scales_selected={}
 for role in ('tail','down'):
  n,k=state['weights'][role]['N'],state['weights'][role]['K'];cols=np.arange(n) if role=='tail' else np.unique(np.linspace(0,n-1,32).astype(np.int64))
  if role=='tail':weight=words_to_float(round_words(np.sin(np.arange(n*k,dtype=np.float32)*.19).reshape(n,k)*.1));weight[0]=0
  else:
   name='lm_head.weight' if role=='lm_head' else 'model.layers.0.mlp.down_proj.weight';d=header[name];assert d['shape']==[n,k];weight=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+d['data_offsets'][0],shape=(n,k))
  original=weight if role=='tail' else words_to_float(np.asarray(weight[cols]));original_selected[role]=original.copy() if role=='tail' else original
  q8,s8,_=independently_quantize(original_selected[role],'W8_channel');base8[role]=(q8,s8[:,0])
  for label in ('W4_sym128','W4_aff128'):
   hs=[hashlib.sha256() for _ in range(3)];selected=np.empty((len(cols),k),np.float32)
   for start in range(0,n,128):
    w=weight[start:start+128] if role=='tail' else words_to_float(weight[start:start+128]);centered,group_scales,decoded=independently_quantize(w,label)
    for phase in range(3):phasewords=round_words(decoded*(1 if phase==0 else -1)*(.5 if phase==2 else 1.));phasewords[(phasewords&0x7fff)==0]=0;hs[phase].update(phasewords.tobytes())
    ids=np.flatnonzero((cols>=start)&(cols<start+len(w)));selected[ids]=centered[cols[ids]-start].astype(np.float32)*group_scales[cols[ids]-start][:,np.arange(k)//128]
   decoded_selected[(role,label)]=selected;centered_q,weight_sc,_=independently_quantize(original_selected[role],label);quantized_selected[(role,label)]=centered_q;group_scales_selected[(role,label)]=weight_sc
   for phase,h in enumerate(hs):
    record=[d for d in state['decodes'] if (d['role'],d['format'],d['phase'])==(role,label,phase)];assert len(record)==1 and record[0]['elements']==n*k and record[0]['actual_BF16_sha256']==record[0]['expected_BF16_sha256']==h.hexdigest();decode_count+=1
  del weight
 cases=state['cases'];assert len(cases)==48 and all(c['complete'] or c.get('compile_failed') for c in cases)
 expected_keys={(role,label,m,t) for role in ('tail','down') for label in ('W4_sym128','W4_aff128') for m in ((17,) if role=='tail' else (64,)) for t in range(12)}
 assert {(c['role'],c['format'],c['M'],c['tile_id']) for c in cases}==expected_keys
 checked=0;partial_integer_elements=0;selected_dots=0;full_cast=0;worst_scaled_error=0.;summary=[]
 for c in cases:
  if c.get('compile_failed'):continue
  role,label,m,n,k=c['role'],c['format'],c['M'],c['N'],c['K'];assert len(c['phases'])==3
  for phase,record in enumerate(c['phases']):
   assert record['phase']==phase and sha(root/record['artifact'])==record['sha256']
   with np.load(root/record['artifact'],allow_pickle=False) as a:
    activation=words_to_float(a['A_words']);assert activation.shape==(m,k)
    rows,cols=a['rows'],a['columns'];wantedcols=np.arange(n) if role=='tail' else np.unique(np.linspace(0,n-1,32).astype(np.int64));assert np.array_equal(cols,wantedcols)
    assert a['full_FP32'].shape==a['full_BF16_words'].shape==(m,n) and np.isfinite(a['full_FP32']).all();assert np.array_equal(round_words(a['full_FP32']),a['full_BF16_words']);full_cast+=m*n
    rng=np.random.default_rng(202610220000+m+k);expected=words_to_float(round_words(rng.standard_normal((m,k),dtype=np.float32)));expected[0]=0;expected[1]=0;expected[1,:9]=[0,.5,1.5,2.5,-.5,-1.5,-2.5,127,-127];expected*=1 if phase==0 else .5 if phase==1 else -.5;assert np.array_equal(activation,expected)
    sign=1 if phase==0 else -1;factor=.5 if phase==2 else 1.;w=decoded_selected[(role,label)].astype(np.float64)*sign*factor;scale=np.maximum(abs(activation).max(1),np.float32(1.e-8))/np.float32(127);q=np.clip(np.rint(activation/scale[:,None]),-127,127).astype(np.int8);left=q[rows].astype(np.float64)*scale[rows,None];reference=np.zeros((len(rows),len(cols)),np.float64)
    bk=c['configuration']['bk'];partials=a['W4_partial_INT32_selected'];assert partials.shape==((k+bk-1)//bk,len(rows),len(cols))
    for block in range((k+bk-1)//bk):
     sl=slice(block*bk,min((block+1)*bk,k));integer_ref=q[rows,sl].astype(np.int64)@(quantized_selected[(role,label)][:,sl].astype(np.int64)*sign).T;assert np.array_equal(partials[block].astype(np.int64),integer_ref);partial_integer_elements+=integer_ref.size;reference+=integer_ref.astype(np.float64)*group_scales_selected[(role,label)][:,block*bk//128][None,:]*factor
    reference*=scale[rows,None];bound=(abs(left)@abs(w).T)*4.e-6+5.e-5;actual=a['full_FP32'][np.ix_(rows,cols)];error=abs(actual-reference);assert np.all(error<=bound),(role,label,m,phase,float(error.max()),float(bound.min()));worst_scaled_error=max(worst_scaled_error,float(np.max(error/bound)));selected_dots+=reference.size
    rawref=activation[rows].astype(np.float64)@(original_selected[role].astype(np.float64)*sign*factor).T;assert np.all(abs(a['native_selected']-rawref)<=(.02+.01*abs(rawref)))
    scale=np.maximum(abs(activation).max(1),np.float32(1.e-8))/np.float32(127);q=np.clip(np.rint(activation/scale[:,None]),-127,127).astype(np.int8);assert np.array_equal(q,a['Aq']) and np.array_equal(scale,a['As'])
    w8,ws8=base8[role];iref=q[rows].astype(np.int64)@(w8.astype(np.int64)*sign).T;assert np.array_equal(iref,a['W8A8_integer_selected'])
    fref=(iref.astype(np.float32)*scale[rows,None])*(ws8[None,:]*factor);bfref=words_to_float(round_words(fref));assert np.array_equal(bfref,a['W8A8_output_selected'])
   checked+=1
  rounds=[r for r in state['timings'] if (r['role'],r['format'],r['M'])==(role,label,m)];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7))
  key=f'tile{c["tile_id"]}';times={kind:float(np.median([r['milliseconds'][kind] for r in rounds])) for kind in ('BF16','W8A8',key)};assert all(v>0 and math.isfinite(v) for v in times.values())
  assert all(r['replays']==20 and set(r['order'])==set(r['milliseconds']) for r in rounds)
  summary.append(dict(role=role,format=label,M=m,tile_id=c['tile_id'],BF16_ms=times['BF16'],W8A8_ms=times['W8A8'],candidate_ms=times[key],speedup_vs_BF16_pct=(times['BF16']/times[key]-1)*100,speedup_vs_W8A8_pct=(times['W8A8']/times[key]-1)*100))
 assert all(sha(p)==v for p,v in state['frozen_sources'].items())
 assert not any(x.split('.')[0] in ('torch','triton','vllm','flag_gems') for x in sys.modules)
 result=dict(complete=True,CPU_only=True,checkpoint_unchanged=True,sources_unchanged=True,independent_full_decode_phases=decode_count,independent_selected_FP64_outputs=selected_dots,full_BF16_FP32_cast_elements=full_cast,phase_artifacts_verified=checked,selected_W4_INT32_partials_exact=partial_integer_elements,worst_selected_FP64_error_over_bound=worst_scaled_error,all_input_quantization_exact=True,W8A8_selected_INT32_and_BF16_exact=True,model_quality_proven=False,official_benchmark=False,adopted=False,raw_directory=str(root),raw_result_sha256=sha(root/'result.json'))
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)
 result['key_timings_sha256']=sha(BASE/'KEY_TIMINGS_32g.csv');(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
