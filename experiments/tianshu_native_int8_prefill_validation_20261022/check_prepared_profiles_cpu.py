# SPDX-License-Identifier: Apache-2.0
import argparse,json,sys
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE))
import quality_profile as q
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
assert not torch.cuda.is_initialized();arms={}
for batch in (8,64):
    arm=root/str(batch);arm.mkdir();command,environment,profile,sources=q.prepare(arm,q.SCREEN,batch=batch)
    full={**profile,'TRITON_CACHE_DIR':environment['TRITON_CACHE_DIR']};assert len(full)==63 and len(profile)==62 and len(sources)==877
    normalized=q.normalized_environment(profile,arm);arms[batch]=dict(command=command,full=full,normalized=normalized,sources=sources)
a,b=arms[8],arms[64];assert a['command']==b['command'] and a['sources']==b['sources']
assert {k for k in a['normalized'] if a['normalized'][k]!=b['normalized'][k]}=={'TIANSHU_FUTURE_KV_MODE'}
assert a['normalized']['TIANSHU_FUTURE_KV_MODE']=='candidate' and b['normalized']['TIANSHU_FUTURE_KV_MODE']=='control'
original=q.read(q.EXP/'tianshu_m_range_prefill_release_20261021/serve_profile_template.json')['frozen_sources'];assert len(original)==471 and all(a['sources'][k]==v for k,v in original.items())
assert q.q64.parent is q.aux.dense and not torch.cuda.is_initialized()
report=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,actual_passing_screen_required=True,runtime_source_count=877,original471_exact_subset=True,same_source_map_both_quality_batches=True,original13_server_argv_identical=True,selected_environment_keys=63,only_normalized_C64_environment_difference='TIANSHU_FUTURE_KV_MODE',old_attention_and_dense_C64_audit_retained=True,source_sha256=q.digest(__file__),screen_gate_sha256=q.digest(q.SCREEN/'SCREEN_GATE_32g.json'),raw_root=str(root))
(BASE/'CPU_PREPARED_GATE_32g.json').write_text(json.dumps(report,indent=2)+'\n');(root/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
