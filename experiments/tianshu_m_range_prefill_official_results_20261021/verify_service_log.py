# SPDX-License-Identifier: Apache-2.0
"""Ready-epoch log and raw preemption metrics, independent of profile flags."""
import hashlib,json,re
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261021_m_range_prefill_resume_official4k_tianshu')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
text=(ROOT/'server.log').read_text();marker='Application startup complete.';assert text.count(marker)==1
post=text.split(marker,1)[1];tuning=[line for line in post.splitlines() if re.search(r'autotun|tuning.*config|config.*tuning|tuned.*shape',line,re.I)];assert not tuning
metrics={}
for name in ('before_metrics.txt','after_metrics.txt'):
 values=re.findall(r'^vllm:num_preemptions_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$',(ROOT/name).read_text(),re.M)
 assert values and all(float(x)==0 for x in values);metrics[name]=dict(values=values,sha256=sha(ROOT/name))
record=json.loads(next((ROOT/'m_range_prefill_records').glob('route_*.json')).read_text());fused=[r for r in record['routes'].values() if r['stage']=='active' and not r['capturing'] and r['implementation']=='M_range_two_dot_fused']
assert len({r['prefix'] for r in fused if r['M']==4096})==42
out=dict(complete=True,CPU_only=True,server_log_sha256=sha(ROOT/'server.log'),ready_marker_count=1,post_ready_autotune_records=0,pattern='autotun|tuning.*config|config.*tuning|tuned.*shape (case insensitive)',all_raw_preemption_metrics_zero=True,metrics=metrics,actual_fused_M_values=sorted({r['M'] for r in fused}),M4096_actual_fused_layers=42,Python_route_counts_are_not_graph_replay_counts=True,raw_root=str(ROOT),verifier_sha256=sha(__file__))
(BASE/'POST_READY_LOG_AUDIT_32g.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
