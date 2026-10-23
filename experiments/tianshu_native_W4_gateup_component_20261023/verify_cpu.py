# SPDX-License-Identifier: Apache-2.0
import csv,hashlib,json,statistics,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def f32(a):return (np.asarray(a,np.uint16).astype(np.uint32)<<16).view(np.float32)
def check(a,r):
 a=f32(a).astype(np.float64);d=a-r;rms=float(np.sqrt(np.mean(r*r)));rel=float(np.linalg.norm(d)/max(np.linalg.norm(r),1.e-30));bad=int(np.count_nonzero(abs(d)>.012*abs(r)+.006*rms));return dict(passed=bool(np.isfinite(a).all() and np.isfinite(r).all() and rel<=.006 and bad==0),relative_L2=rel,failed_cells=bad,cells=a.size)
def main():
 g=json.loads((BASE/'CPU_GATE_32g.json').read_text());root=Path(g['root']);out=root/'GPU';s=json.loads((out/'result.json').read_text());assert s['complete'] and s['phase']=='complete' and s['all_input_full_bytes_versions_preserved'] and s['RNG_unchanged'];assert all(sha(p)==h for p,h in g['sources'].items()) and all(sha(p)==h for p,h in g['files'].items());checks=[]
 for row in s['numeric']:
  assert sha(row['file'])==row['sha256'] and row['eager_graph_and_repeat_exact'];a=dict(np.load(row['file'],allow_pickle=False));ref=dict(np.load(root/f"ref_L{row['layer']}_p{row['phase']}.npz",allow_pickle=False));candidate=row['variant']=='native_W4_plus_same_SiLU';r=check(a['output'],ref['candidate' if candidate else 'control']);checks.append(dict(layer=row['layer'],phase=row['phase'],variant=row['variant'],kind='activated',**r))
  if candidate:checks.append(dict(layer=row['layer'],phase=row['phase'],variant=row['variant'],kind='projection',**check(a['projection'],ref['projection4'])))
 eligible=all(c['passed'] for c in checks);rows=[]
 for layer in g['layers']:
  for mode in ('eager','graph'):
   selected=[r for r in s['rounds'] if r['layer']==layer and r['mode']==mode and not r['excluded']];assert len(selected)==6
   control=[r['ms']['r4_W8A8_two_dot_SiLU'] for r in selected];candidate=[r['ms']['native_W4_plus_same_SiLU'] for r in selected];c,v=statistics.median(control),statistics.median(candidate);rows.append(dict(layer=layer,mode=mode,control_ms=c,candidate_ms=v,speedup_percent=100*(c/v-1),candidate_hot_spread_percent=100*(max(candidate)-min(candidate))/statistics.mean(candidate),numeric_qualified=eligible))
 with (BASE/'KEY_TIMINGS_32g.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 gate=dict(complete=True,CPU_only=True,numeric_qualified=eligible,all27_full_FP64_output_checks_passed=eligible,checks=checks,timings=rows,raw_root=str(root),raw_result_sha256=sha(out/'result.json'),CPU_gate_sha256=sha(BASE/'CPU_GATE_32g.json'),key_timings_sha256=sha(BASE/'KEY_TIMINGS_32g.csv'),baseline_real_r4_W8_weights_and_selected_kernel=True,actual_model_quality_not_tested=True,official_benchmark=False,adopted=False)
 assert len(checks)==27 and not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);(BASE/'INDEPENDENT_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='checks'}));assert eligible
if __name__=='__main__':main()
