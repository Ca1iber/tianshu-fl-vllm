# SPDX-License-Identifier: Apache-2.0
import ast,json,os
os.environ['VLLM_PLUGINS']=''
from pathlib import Path
import torch
from candidate_profile import BASE,prepare,read,digest
assert not torch.cuda.is_initialized()
for p in BASE.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
root=Path('/workspace/benchmark-results/20261021_nn_silu_fx_screen_CPU_tianshu');assert not root.exists();root.mkdir()
old=(BASE.parent/'tianshu_prefill_int8_service_20261021/run_service_trial.py').read_text()
original_block="""            for batch in (8,64):
                assert len({r['bank_id'] for r in binding['launches'].values() if r['role']=='gateup' and r['M']==batch and r['actually_inside_capture']})==42, ('early_missing_GateUp',batch)"""
new_block='''            for batch in (8,64):
                if args.mode=="candidate" and batch==64:
                    actual=read(next((root/"nn_silu_fx_records").glob("route_*.json")))
                    assert len({r["bank_id"] for r in actual["routes"].values() if r["M"]==64 and r["actually_inside_capture"]})==42
                else:
                    assert len({r["bank_id"] for r in binding["launches"].values() if r["role"]=="gateup" and r["M"]==batch and r["actually_inside_capture"]})==42'''
assert original_block in old and (BASE/'run_service_trial.py').read_text()==old.replace(original_block,new_block)
assert [(length,m) for length in (63,64) for m in (32,4001)]==[(63,32),(63,4001),(64,32),(64,4001)]
for length in (63,64):
    source=torch.arange(length,dtype=torch.int32)*3+7
    for m in (32,4001):
        idx=torch.arange(m,dtype=torch.int64)%length;expected=source[idx]
        assert torch.equal(expected,3*idx.to(torch.int32)+7)
profiles=[]
for mode in ('control','candidate'):
    target=root/mode;target.mkdir();command,environment,profile,sources=prepare(target,mode);profiles.append((command,profile,sources))
assert profiles[0][0]==profiles[1][0] and profiles[0][2]==profiles[1][2]
normalized=[{k:v.replace(str(root/mode),'{trial}') for k,v in row[1].items() if k!='PYTHONPATH'} for mode,row in zip(('control','candidate'),profiles)];assert normalized[0]==normalized[1]
source=(BASE/'dense_audit.py').read_text();assert source.count('for launch in actual_gateup_launches(root,row,state,batch):')==1
assert "lm=[r for r in row['launches'].values()" in source
assert "else:assert sum(r[\"calls\"] for r in lm)>=replays" in source
assert not torch.cuda.is_initialized()
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,GPU_not_executed=True,benchmark_sampling_and_request_timing_unmodified=True,only_early_actual_M64_evidence_check_adapted=True,
          exact_four_observed_index_warm_shapes_CPU_references_verified=True,mode_environment_only_hook_and_output_directory_differences=True,control_candidate_sources_identical=True,
          original407_reference_uses_original_service_entry=True,source_count=len(profiles[0][2]),sources=profiles[0][2],controllers={str(p):digest(p) for p in [*BASE.rglob('*.py'),BASE/'PLAN_32g.md',BASE/'run_screen.sh']})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','controllers')}))
