# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,math,statistics,struct,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors');RAW=Path('/workspace/benchmark-results/20261022_down_group_GPU_tianshu')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def f32(a):return (np.asarray(a,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16(a):
 u=np.asarray(a,np.float32).view(np.uint32);return ((u+32767+((u>>16)&1))>>16).astype(np.uint16)
def quantize(a):
 s=np.asarray(np.maximum(abs(a).max(1),np.float32(1.e-8)).astype(np.float64)/127.,np.float32);return np.clip(np.rint(np.asarray(a.astype(np.float64)/s[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8),s

def main():
 assert not (BASE/'INDEPENDENT_CPU_GATE_32g.json').exists();state=read(RAW/'result.json');assert state['complete'] and state['phase']=='complete' and state['checkpoint_unchanged'] and state['all_bank_hashes_before_after'] and state['single_bank_cache_screen_only'] and [c['M'] for c in state['cases']]==[64,512,4096]
 for p,h in state['frozen_sources'].items():assert sha(p)==h and sha(RAW/'source_snapshot'/Path(p).relative_to('/'))==h
 assert sha(MODEL)==state['checkpoint_sha256_before']==state['checkpoint_sha256_after']=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
 with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 info=header['model.layers.0.mlp.down_proj.weight'];assert info['shape']==[2048,6144] and info['dtype']=='BF16';ww=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(2048,6144));weight=f32(ww);packed,ws=quantize(weight);bank=state['bank'];assert ahash(ww)==bank['Down_original_BF16_sha256'] and ahash(packed)==bank['Down_packed_sha256'] and ahash(ws)==bank['Down_scales_sha256'];artifacts=integer_cells=projection_cells=0
 for case in state['cases']:
  assert case['complete'] and len(case['phases'])==2
  for phase in case['phases']:
   assert phase['seed']==202610225800+case['M']*10+phase['phase'] and phase['full_grouped_BF16_equals_native_group_INT32_FP32_reference'] and phase['graph_poison_eager_exact'] and phase['guards_inputs_weights_RNG'];path=RAW/phase['artifact'];assert sha(path)==phase['sha256']
   with np.load(path,allow_pickle=False) as data:
    h=f32(data['H_selected_rows_words']);cols=data['columns'];assert h.shape==(8,6144) and cols.shape==(32,) and not h[0].any()
    aq,asc=quantize(h);integer=aq.astype(np.int64)@packed[cols].astype(np.int64).T;assert np.array_equal(integer,data['ordinary_integer_selected']) and np.array_equal(bf16((integer.astype(np.float32)*asc[:,None])*ws[None,cols]),data['ordinary_selected_words'])
    for group in (128,512):
     codes=np.empty_like(h,np.int8);scales=np.empty((8,6144//group),np.float32);acc=np.zeros((8,32),np.float32)
     for index,start in enumerate(range(0,6144,group)):
      q,s=quantize(h[:,start:start+group]);codes[:,start:start+group]=q;scales[:,index]=s;integer=q.astype(np.int64)@packed[cols,start:start+group].astype(np.int64).T;assert np.array_equal(integer,data[f'G{group}_integer_groups_selected'][index]);integer_cells+=integer.size;acc=np.asarray(acc+integer.astype(np.float32)*s[:,None],np.float32)
     assert np.array_equal(codes,data[f'G{group}_Aq_rows']) and np.array_equal(scales,data[f'G{group}_AS_rows']);expected=bf16(np.asarray(acc*ws[None,cols],np.float32));assert np.array_equal(expected,data[f'G{group}_reference_selected_words'])
     for warps in (2,4):assert np.array_equal(expected,data[f'G{group}_w{warps}_selected_words']);projection_cells+=expected.size
    ideal=h.astype(np.float64)@weight[cols].astype(np.float64).T;absolute=abs(h.astype(np.float64))@abs(weight[cols].astype(np.float64)).T;gamma=6144*np.finfo(np.float32).eps/(1-6144*np.finfo(np.float32).eps);assert np.all(abs(f32(data['control_selected_words']).astype(np.float64)-ideal)<=.00390625*abs(ideal)+gamma*absolute+1.e-7)
   artifacts+=1
 assert artifacts==6 and integer_cells==92160 and projection_cells==6144;table=[]
 for m in (64,512,4096):
  for scope in ('whole_MLP','Down_only'):
   for mode in ('graph','eager'):
    rows=[r for r in state['rounds'] if r['M']==m and r['scope']==scope and r['mode']==mode];assert len(rows)==7 and sorted(r['round'] for r in rows)==list(range(7));names=list(rows[0]['milliseconds']);values={name:statistics.median(row['milliseconds'][name] for row in rows) for name in names};assert all(math.isfinite(v) and v>0 for v in values.values())
    for name,value in values.items():table.append(dict(M=m,scope=scope,mode=mode,variant=name,median_ms=value,speedup_vs_BF16_pct=100*(values['control']/value-1),speedup_vs_ordinary_native_INT8_pct=100*(values['ordinary_INT8']/value-1)))
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
 result=dict(complete=True,CPU_only=True,no_GPU_imports=True,single_bank_cache_screen_only=True,independent_random_inputs=True,independent_INT64_group_dot_cells=integer_cells,independent_exact_BF16_projection_cells=projection_cells,full_GPU_candidates_equal_independent_native_group_reference=True,all_source_snapshots_and_checkpoint_verified=True,graph_poison_eager_and_canaries_verified=True,artifacts_checked=artifacts,numerical_full_BF16_error_by_case={str(c['M']):[p['relative_L2_vs_original_BF16'] for p in c['phases']] for c in state['cases']},timings=table,full42_bank_holdout_still_required=True,new_full_C8_C64_Math_required=True,model_executed=False,official_benchmark=False,adopted=False,raw_directory=str(RAW),raw_result_sha256=sha(RAW/'result.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'))
 (BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps(result))
if __name__=='__main__':main()
