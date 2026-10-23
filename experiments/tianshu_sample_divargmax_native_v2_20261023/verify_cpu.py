# SPDX-License-Identifier: Apache-2.0
"""Independent complete saved-output audit using NumPy and native loop rules."""
import hashlib,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;root=Path('/dev/shm/tianshu_sample_divargmax_native_v2_20261023');p=root/'result.json';s=json.loads(p.read_text());assert s['phase']=='complete' and s['all_GPU_checks_passed'] and len(s['cases'])==16
for name,h in s['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
rows=[];cells=0
for entry in s['cases']:
 f=Path(entry['file']);assert hashlib.sha256(f.read_bytes()).hexdigest()==entry['sha256'];z=np.load(f);probs=z['probs_bits'].view(np.float32);noise=z['noise_bits'].view(np.float32);native=z['native_score_bits'];candidate=z['candidate_score_bits'];assert np.array_equal(native,candidate)
 with np.errstate(all='ignore'):expected=np.divide(probs,noise,dtype=np.float32)
 observed=candidate.view(np.float32);finite=np.isfinite(expected);assert np.array_equal(np.isnan(expected),np.isnan(observed));assert np.array_equal(np.isposinf(expected),np.isposinf(observed));assert np.array_equal(np.isneginf(expected),np.isneginf(observed))
 denominator=np.maximum(np.abs(expected[finite]).astype(np.float64),np.finfo(np.float32).tiny);relative=np.abs(expected[finite].astype(np.float64)-observed[finite].astype(np.float64))/denominator;max_relative=float(relative.max(initial=0));assert max_relative<=2e-6
 b,n=expected.shape;tile=(1<<(n-1).bit_length()) if n<=32768 else 4096;tokens=[]
 for row in expected:
  if n<=tile:tokens.append(int(np.argmax(row)));continue
  maximum=np.finfo(np.float32).min;index=0
  for start in range(0,n,tile):
   part=row[start:start+tile];local=np.max(part)
   # This is the actual native loop behavior: NaN > max is false.
   if local>maximum:maximum=local;index=start+int(np.argmax(part))
  tokens.append(index)
 tokens=np.asarray(tokens,dtype=np.int64);assert np.array_equal(tokens,z['native_tokens']) and np.array_equal(tokens,z['candidate_tokens']);cells+=observed.size
 rows.append(dict(case=entry['case'],phase=entry['phase'],batch=b,vocab=n,pattern=entry['pattern'],full_scores_native_candidate_bitwise_equal=True,CPU_FP32_division_relative_error_max=max_relative,CPU_native_loop_token_exact=True,standard_NumPy_argmax_disagreeing_rows=int(np.count_nonzero(np.argmax(expected,axis=-1)!=tokens)),native_tile_n=tile))
gate=dict(complete=True,component_numerics_qualified=True,model_quality_qualified=False,adopted=False,official_benchmark=False,raw_directory=str(root),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),full_FP32_cells=cells,all_recorded_source_SHA_verified=True,cases=rows,timings=s['timings'],verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),initial_custom_NaN_rule_failure_not_reclassified=True,initial_native_launch_binding_failure_not_reclassified=True)
(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(dict(complete=True,cells=cells,timings=s['timings'],NaN_standard_argmax_differences=sum(r['standard_NumPy_argmax_disagreeing_rows'] for r in rows))))
