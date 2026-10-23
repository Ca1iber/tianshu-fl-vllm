# SPDX-License-Identifier: Apache-2.0
import csv,sys
from common import *
g=json.loads((BASE/'CPU_GATE_32g.json').read_text());s=json.loads((GPU/'result.json').read_text());assert s['complete'] and s['checkpoint_sources_and_inputs_unchanged'] and s['all_guards_versions_full_bytes_preserved'] and s['RNG_unchanged'] and sha(MODEL)==g['checkpoint_sha256']==s['checkpoint_sha256_after'];assert all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items());assert len(s['numeric'])==9;checks=[]
for r in s['numeric']:
 assert r['full_graph_poison_eager_exact'] and r['all_output_finite'] and sha(r['file'])==r['sha256'];actual=np.load(r['file'],allow_pickle=False);assert actual.shape==(M,N);phase=r['phase'];original=np.load(CPU/f'original_FP64_{phase}.npy',mmap_mode='r');av=f32(actual).astype(np.float64);quantization_error=float(np.linalg.norm(av-original)/max(np.linalg.norm(original),1e-30))
 if r['variant']=='W8A8_current':
  assert sha(r['A8_file'])==r['A8_sha256'];a=np.load(r['A8_file']);x=np.load(CPU/f'X{phase}.npy');q,asc=quant_a(x);assert np.array_equal(q,a['Q']) and np.array_equal(asc,a['S']);w8=np.load(CPU/'W8.npy',mmap_mode='r');s8=np.load(CPU/'S8.npy');integer=q.astype(np.int64)@w8[COLS].astype(np.int64).T;expected=bf16((integer.astype(np.float32)*asc[:,None])*s8[COLS][None,:]);assert np.array_equal(integer,np.load(CPU/f'INT64_selected{phase}.npy')) and np.array_equal(expected,np.load(CPU/f'W8_exact_selected{phase}.npy'));different=int(np.count_nonzero(actual[:,COLS]!=expected));check=dict(passed=different==0,different_cells=different,cells=int(expected.size),full_A8_and_scale_exact=True,reference='full_K_INT64_selected_columns_BF16_exact')
 else:check=dict(**compare(actual,np.load(CPU/f'quantized_FP64_{phase}.npy',mmap_mode='r')),reference='all_output_full_K_FP64_of_BF16_dequantized_weights')
 checks.append(dict(variant=r['variant'],phase=phase,quantization_relative_L2_vs_original_FP64=quantization_error,**check))
rows=[]
for mode in ('eager','graph'):
 rounds=[r for r in s['rounds'] if r['mode']==mode];assert len(rounds)==7 and {r['round'] for r in rounds}==set(range(7))
 for r in rounds:assert r['excluded']==(r['round']==0) and r['replays']==20 and set(r['order'])==set(VARIANTS) and set(r['milliseconds'])==set(VARIANTS)
 med={n:float(np.median([r['milliseconds'][n] for r in rounds if r['round']>0])) for n in VARIANTS};assert all(np.isfinite(v) and v>0 for v in med.values())
 for name in VARIANTS[1:]:rows.append(dict(mode=mode,variant=name,control_ms=med[VARIANTS[0]],candidate_ms=med[name],time_saved_ms=med[VARIANTS[0]]-med[name],speedup_pct=100*(med[VARIANTS[0]]/med[name]-1)))
with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
assert not any(k.split('.')[0] in ('torch','triton','vllm','flag_gems') for k in sys.modules)
r=dict(complete=True,CPU_only=True,all_numeric_passed=all(c['passed'] for c in checks),checks=checks,timings=rows,CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),result_sha256=sha(GPU/'result.json'),raw_directory=str(GPU),adopted=False,model_quality_not_tested=True,official_benchmark=False);(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
