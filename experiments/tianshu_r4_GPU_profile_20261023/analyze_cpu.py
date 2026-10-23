# SPDX-License-Identifier: Apache-2.0
import collections,hashlib,json,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent;root=Path('/workspace/benchmark-results/20261023_r4_GPU_profile_tianshu');s=json.loads((root/'result.json').read_text());assert s['complete'] and s['phase']=='complete'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for p,h in s['frozen_sources'].items():assert sha(p)==h and sha(root/'source_snapshot'/Path(p).relative_to('/'))==h
r=json.loads(Path(s['GPU_profile_record']).read_text());assert sha(s['GPU_trace_file'])==r['trace_sha256'];es=json.loads(Path(s['GPU_trace_file']).read_text())['traceEvents'];groups=collections.defaultdict(list)
for e in es:
 if e.get('cat')=='kernel':groups[e['name']].append(e['dur'])
kernels=[dict(name=n,count=len(ds),total_ms=sum(ds)/1000,median_us=statistics.median(ds)) for n,ds in sorted(groups.items(),key=lambda p:sum(p[1]),reverse=True)]
metrics={}
for name in ('before_reused_salt','after_reused_salt','after_fresh_salt'):
 rows=(root/(name+'_metrics.txt')).read_text().splitlines();metrics[name]={k:float(next(l.split()[-1] for l in rows if l.startswith('vllm:'+k+'{'))) for k in ('prefix_cache_hits_total','prefix_cache_queries_total')}
assert metrics['after_reused_salt']['prefix_cache_hits_total']-metrics['before_reused_salt']['prefix_cache_hits_total']==112
assert metrics['after_fresh_salt']['prefix_cache_hits_total']==metrics['after_reused_salt']['prefix_cache_hits_total']
assert s['same_seed_cold_cache_text_equal'] and not s['same_seed_reused_salt_text_equal']
result=dict(complete=True,diagnostic_only=True,adopted=False,model_quality_qualified=False,official_benchmark=False,source_count=len(s['frozen_sources']),original_model_source_count=len(s['original_model_sources']),all_current_and_captured_source_SHA_verified=True,frames=r['frame_count'],kernel_events=r['GPU_kernel_events'],trace_SHA256=r['trace_sha256'],kernels=kernels,CUDA_Graph_child_kernels_not_observed=True,CUDA_Graph_launch_count=sum(e.get('cat')=='cuda_runtime' and e.get('name')=='cudaGraphLaunch' for e in es),no_total_model_GPU_breakdown_claimed=True,cache_metrics=metrics,canaries={k:s[k] for k in ('canary_cold_before','canary_reused_salt_after','canary_cold_after')},same_seed_cold_cache_text_equal=True,same_seed_reused_salt_text_equal=False,prior_CPU_profile_failed_canary_not_reclassified=True,raw_result_sha256=sha(root/'result.json'))
(BASE/'KEY_GPU_PROFILE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('kernels','canaries','cache_metrics')},indent=2))
