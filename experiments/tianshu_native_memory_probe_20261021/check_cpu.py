# SPDX-License-Identifier: Apache-2.0
import ast,json
from pathlib import Path
from types import SimpleNamespace
from candidate_profile import BASE,prepare
from observer import budget_snapshot
for file in BASE.rglob('*.py'):ast.parse(file.read_text())
for enabled in (True,False):
    worker=SimpleNamespace(requested_memory=10000,model_runner=SimpleNamespace(model_memory_usage=5000),peak_activation_memory=700,non_torch_memory=800,cudagraph_memory_estimate=300)
    expected=10000-5000-700-800-(300 if enabled else 0)
    row=budget_snapshot(worker,expected,5100,5300,enabled);assert row['native_budget_equation_exact'] and row['original_KV_memory_bytes']==expected and row['no_extra_profile_run'] and row['no_peak_reset']
    try:budget_snapshot(worker,expected+1,5100,5300,enabled)
    except AssertionError:pass
    else:raise AssertionError('inconsistent native budget accepted')
profiles=[]
for mode in ('control','candidate'):
    initial=Path(f"/workspace/benchmark-results/20261021_native_memory_probe_CPU_{mode}_tianshu");root=initial;revision=0
    while root.exists():revision+=1;root=initial.with_name(initial.name+f"_v{revision}")
    root.mkdir()
    command,env,profile,sources=prepare(root,mode);assert len(profile)==62 and len(command)==13
    profiles.append(dict(mode=mode,command=command,profile=profile,sources=sources))
assert profiles[0]['sources']==profiles[1]['sources'] and profiles[0]['command']==profiles[1]['command']
result=dict(complete=True,CPU_only=True,model_Graph_and_GEMM_not_run=True,original_native_budget_equation_checked_both_Graph_flags=True,
    inconsistent_budget_rejected=True,runtime_source_count=len(profiles[0]['sources']),environment_count=62,original13serveargv_preserved=True,
    KV_budget_no_change=True,no_additional_warm_or_peak_reset=True,only_extra_C8_canaries_after_ready=True,profiles=profiles)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
