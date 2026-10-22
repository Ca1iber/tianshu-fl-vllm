# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;EXP=BASE.parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');ROOT=Path('/workspace/benchmark-results/20261022_attention_weight_groups_CPU_tianshu')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def f32(words):return (np.asarray(words,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16(v):
 bits=np.asarray(v,np.float32).view(np.uint32);return ((bits+32767+((bits>>16)&1))>>16).astype(np.uint16)
def pack(v):
 s=np.asarray(np.maximum(abs(v).max(1),np.float32(1e-8)).astype(np.float64)/127.,np.float32);return np.clip(np.rint(np.asarray(v.astype(np.float64)/s[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8),s

def main():
 assert not ROOT.exists() and not (BASE/'CPU_GATE_32g.json').exists();ROOT.mkdir();before=sha(MODEL);assert before=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c';sources={str(p):sha(p) for p in (BASE/'analyze_cpu.py',BASE/'PLAN_32g.md')}
 for p,h in sources.items():target=ROOT/Path(p).name;target.write_bytes(Path(p).read_bytes());assert sha(target)==h
 with MODEL.open('rb') as stream:size=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(size))
 def get(layer,role):
  e=header[f'model.layers.{layer}.self_attn.{role}_proj.weight'];assert e['dtype']=='BF16';return f32(np.array(np.memmap(MODEL,mode='r',dtype='<u2',offset=8+size+e['data_offsets'][0],shape=tuple(e['shape']))))
 fixtures=[]
 for phase in range(2):
  values=np.random.default_rng(202610235000+phase).standard_normal((32,2048),dtype=np.float32);values=f32(bf16(values));values[0]=0;aq,asc=pack(values);fixtures.append((values,aq,asc))
 old_gate=json.loads((EXP/'tianshu_native_int8_attention_projections_20261022/INDEPENDENT_CPU_GATE_32g.json').read_text());old_root=Path(old_gate['raw_directory']);old=json.loads((old_root/'result.json').read_text());assert old_gate['complete'] and sha(old_root/'result.json')==old_gate['raw_result_sha256'];records=[];old_cells=0
 for layer in range(42):
  for role in ('QKV','O'):
   w=np.concatenate([get(layer,p) for p in ('q','k','v')],0) if role=='QKV' else get(layer,'o');n,k=w.shape;code,scale=pack(w);columns=np.unique(np.concatenate([np.linspace(0,2047,40).astype(np.int64),np.linspace(2048,n-1,24).astype(np.int64)])) if role=='QKV' else np.unique(np.linspace(0,n-1,64).astype(np.int64));chosen=w[columns];wd=chosen.astype(np.float64);qd=code[columns];sc=scale[columns]
   if layer==0:
    case=next(c for c in old['cases'] if c['role']==role and c['M']==128);signature=case['weight_signature'];assert ahash(bf16(w))==signature['original_BF16_sha256'] and ahash(code)==signature['packed_sha256'] and ahash(scale)==signature['scales_sha256'];p=case['phases'][0];assert sha(old_root/p['artifact'])==p['sha256']
    with np.load(old_root/p['artifact']) as a:
     x=f32(a['X_words']);aq,asc=pack(x);integer=aq[a['rows']].astype(np.int64)@code[a['columns']].astype(np.int64).T;assert np.array_equal(integer,a['integer_selected']) and np.array_equal(bf16((integer.astype(np.float32)*asc[a['rows'],None])*scale[None,a['columns']]),a['candidate_selected_words']);old_cells+=integer.size
   group_banks={}
   for group in (128,512):
    grouped=np.empty_like(chosen,np.int8);scales=np.empty((len(columns),2048//group),np.float32)
    for index,start in enumerate(range(0,k,group)):grouped[:,start:start+group],scales[:,index]=pack(chosen[:,start:start+group])
    group_banks[group]=(grouped,scales)
   for phase,(x,aq,asc) in enumerate(fixtures):
    original=x.astype(np.float64)@wd.T;integer=aq.astype(np.int64)@qd.astype(np.int64).T;baseline=f32(bf16((integer.astype(np.float32)*asc[:,None])*sc[None,:])).astype(np.float64);den=float(np.sum(original*original,dtype=np.float64));errors={'per_output':float(np.sum((baseline-original)**2,dtype=np.float64))}
    for group,(grouped,scales) in group_banks.items():
     total=np.zeros_like(original)
     for index,start in enumerate(range(0,k,group)):
      partial=aq[:,start:start+group].astype(np.int64)@grouped[:,start:start+group].astype(np.int64).T;total+=partial.astype(np.float64)*scales[None,:,index].astype(np.float64)
     out=f32(bf16(total*asc[:,None].astype(np.float64))).astype(np.float64);errors[f'G{group}_ideal']=float(np.sum((out-original)**2,dtype=np.float64))
    for name,numerator in errors.items():records.append(dict(layer=layer,role=role,phase=phase,variant=name,output_columns=len(columns),squared_error=numerator,squared_reference=den,relative_L2=(numerator/den)**.5))
  print(json.dumps(dict(layer=layer,complete=True)),flush=True)
 with (BASE/'KEY_ERRORS_32g.csv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 summary={}
 for role in ('QKV','O'):
  summary[role]={name:(sum(r['squared_error'] for r in records if r['role']==role and r['variant']==name)/sum(r['squared_reference'] for r in records if r['role']==role and r['variant']==name))**.5 for name in ('per_output','G128_ideal','G512_ideal')}
 assert before==sha(MODEL) and all(sha(p)==h for p,h in sources.items()) and not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);gate=dict(complete=True,CPU_only=True,no_GPU_imports=True,independent_new_synthetic_fixtures=True,all84_original_banks=True,old_layer0_native_INT64_BF16_selected_cells_verified=old_cells,checkpoint_sha256=before,sources=sources,records=len(records),weighted_relative_L2=summary,grouped_accumulation='ideal_FP64_not_claimed_GPU_FP32_FMA',GPU_tested=False,model_tested=False,adopted=False,key_errors_sha256=sha(BASE/'KEY_ERRORS_32g.csv'));(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
