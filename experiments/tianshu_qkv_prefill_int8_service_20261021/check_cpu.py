# SPDX-License-Identifier: Apache-2.0
"""Independent all-checkpoint packing and exact current-profile/source preflight."""
import ast,json,tempfile,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from safetensors import safe_open
from candidate_profile import BASE,prepare,read,digest
from weight_pack import pack,describe,tensor_sha
from routing import configuration,eligible
assert not torch.cuda.is_initialized();torch.set_num_threads(4)
for path in BASE.rglob('*.py'):ast.parse(path.read_text())
selection=read(BASE/'selected_plan.json');gate=read(selection['component_gate'])
assert gate['complete'] and gate['independently_CPU_rechecked'] and gate['original_GPU_run_remains_failed'] and gate['partial_scope']=='completed_QKV_only'
assert digest(selection['component_gate'])==selection['component_gate_sha256'] and digest(selection['kernel_source'])==selection['kernel_sha256']
for path,h in gate['frozen_sources'].items():assert digest(path)==h
selected={r['M']:r['configuration'] for r in selection['configurations']};assert set(selected)=={128,256,512,1024,2048,4096}
components={r['M']:r for r in gate['best_by_case'] if r['case']=='qkv'}
for row in selection['configurations']:
    assert all(row[k]==components[row['M']][k] for k in ('configuration','native_ms','candidate_ms')) and row['candidate_ms']<row['native_ms']
parent=BASE.parent/'tianshu_prefill_int8_service_20261021'
assert (BASE/'run_service_trial.py').read_bytes()==(parent/'run_service_trial.py').read_bytes()
expected_hook=(parent/'hooks_candidate/sitecustomize.py').read_text().replace('tianshu_int8_nn_gateup_service_20261021/hooks_candidate/sitecustomize.py','tianshu_prefill_int8_service_20261021/hooks_candidate/sitecustomize.py').replace('_prefill_int8_service_policy','_qkv_prefill_int8_service_policy').replace("/'prefill_int8_records'","/'qkv_prefill_int8_records'")
for mode in ('control','candidate'):assert (BASE/f'hooks_{mode}/sitecustomize.py').read_text()==expected_hook
before=torch.get_rng_state().clone();banks=[];elements=0
checkpoints=list(Path('/workspace/MiniCPM5-2B').glob('*.safetensors'));assert len(checkpoints)==1
legacy=read(BASE.parent/'tianshu_int8_attention_projections_20261021/weight_signatures.json')['qkv']
with safe_open(str(checkpoints[0]),framework='pt',device='cpu') as checkpoint:
    for layer in range(42):
        prefix=f'model.layers.{layer}.self_attn.';keys=[prefix+s+'_proj.weight' for s in ('q','k','v')]
        weight=torch.cat([checkpoint.get_tensor(key) for key in keys],0).contiguous()
        quant,scale=pack(weight);values=weight.float().numpy()
        scale_np=np.maximum(np.max(np.abs(values),axis=1),np.float32(1.e-8))/np.float32(127.)
        quant_np=np.clip(np.rint(values/scale_np[:,None]),-127,127).astype(np.int8)
        assert torch.equal(quant,torch.from_numpy(quant_np)) and torch.equal(scale,torch.from_numpy(scale_np))
        row=dict(prefix=prefix+'qkv_proj',checkpoint_keys=keys,**describe(weight))
        if layer==0:
            assert row['original_BF16_sha256']==legacy['BF16_bits_sha256'] and row['packed_logical_NK_sha256']==legacy['packed_sha256'] and row['scales_sha256']==legacy['scales_sha256']
        banks.append(row);elements+=weight.numel()
        print(json.dumps(dict(checkpoint_layer=layer,full_weight_and_numpy_quantization_verified=True)),flush=True)
        del weight,quant,scale,values,scale_np,quant_np
assert elements==220200960 and sum(r['payload_bytes'] for r in banks)==220631040
manifest=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,checkpoint=str(checkpoints[0]),checkpoint_file_bytes=checkpoints[0].stat().st_size,
    all_full_QKV_weight_elements=elements,all_42_independent_numpy_quantization_verified=True,layer0_matches_qualified_component=True,banks=banks)
(BASE/'CHECKPOINT_SIGNATURES_32g.json').write_text(json.dumps(manifest,indent=2)+'\n')
configs=0;programs=0;warm={m:configuration(m,selected)[1] for m in range(128,4097,128)}
for m,config in warm.items():
    bm,bn,group=config['bm'],config['bn'],config['group_m'];tm=(m+bm-1)//bm;tn=(2560+bn-1)//bn;pairs=[]
    for pid in range(tm*tn):
        gid=pid//(group*tn);first=gid*group;size=min(tm-first,group);local=pid%(group*tn)
        pairs.append((first+local%size,local//size));programs+=1
    assert len(set(pairs))==tm*tn and set(pairs)=={(i,j) for i in range(tm) for j in range(tn)};configs+=1
class Probe:
    def __init__(self,m,k,ndim,contiguous):self.ndim=ndim;self.shape=(m,k);self.contiguous=contiguous
    def is_contiguous(self):return self.contiguous
routes=0
for m in range(4353):
 for k in (2048,2049):
  for ndim in (2,3):
   for contiguous in (False,True):
    x=Probe(m,k,ndim,contiguous)
    for mode in ('control','candidate'):
     for active in (False,True):
      expected=mode=='candidate' and active and ndim==2 and m in warm and k==2048 and contiguous
      assert eligible(mode,active,x,warm)==expected;routes+=1
profiles=[]
for mode in ('control','candidate'):
    root=Path(tempfile.mkdtemp(prefix=f'20261021_qkv_prefill_CPU_{mode}_',dir='/workspace/benchmark-results'))
    command,environment,profile,sources=prepare(root,mode)
    assert len(profile)==62 and len(command)==13 and profile['PYTHONPATH']==str(BASE/('hooks_'+mode))
    assert all(profile[k]=='candidate' for k in ('TIANSHU_DENSE_INT8_MODE','TIANSHU_LM_PRECISION_MODE','TIANSHU_PREFILL_LINEAR_MODE','TIANSHU_SEQ_LIMIT_MODE'))
    profiles.append(dict(mode=mode,command=command,profile=profile,sources=sources,CPU_profile_root=str(root)))
assert profiles[0]['sources']==profiles[1]['sources'] and profiles[0]['command']==profiles[1]['command']
assert torch.equal(torch.get_rng_state(),before) and not torch.cuda.is_initialized()
result=dict(complete=True,CPU_only=True,no_GPU_started=True,all_42_checkpoint_and_packed_banks_verified=True,full_QKV_weight_elements=elements,
    full_independent_numpy_quantization_elements=elements,expected_extra_payload_bytes=220631040,selected_warm_geometry_checks=configs,launch_tile_pairs=programs,
    eligible_route_cases=routes,default_CPU_RNG_unchanged=True,CUDA_not_initialized=True,parent_407_runtime_sources_preserved=True,
    runtime_source_count=len(profiles[0]['sources']),environment_count=62,original_serve_command_preserved=True,same_current_GateUp_prefill_candidate_both_arms=True,
    all_QKV_decode_delegated_requires_actual_capture_audit=True,new_model_Math_required=True,profiles=profiles)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
