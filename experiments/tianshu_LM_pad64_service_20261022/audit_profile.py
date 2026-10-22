# SPDX-License-Identifier: Apache-2.0
import hashlib,importlib.util,json,sys,types
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def patch_dense_modules(root):
 queue=[root];seen=set();patched=[];target=EXP/'tianshu_lm_precision_service_20261020/dense_audit.py'
 while queue:
  module=queue.pop()
  if id(module) in seen:continue
  seen.add(id(module));path=getattr(module,'__file__',None)
  if not path or not Path(path).is_relative_to(EXP):continue
  queue.extend(v for v in vars(module).values() if isinstance(v,types.ModuleType))
  if Path(path).resolve()==target.resolve():
   name=f'_lm_new_dense_explicit_{len(patched)}';spec=importlib.util.spec_from_file_location(name,BASE/'dense_audit.py');replacement=importlib.util.module_from_spec(spec);sys.modules[name]=replacement;spec.loader.exec_module(replacement);replacement.parent=module.parent;module.audit_runtime=replacement.audit_runtime;patched.append(str(target))
 assert patched;return patched

def audit(root,state):
 files=list((root/'lm_pad64_records').glob('route_*.json'));assert len(files)==1;r=read(files[0]);selection=read(BASE/'selected_plan.json');assert r['installed'] and r['completed'] and r['phase']=='runtime' and r['mode']=='candidate' and r['minimum_M']==selection['minimum_M'] and r['maximum_M']==r['physical_M']==64
 for field in ('source','kernel','selection','original_logits'):
  p=r[field] if field=='source' else r[field+'_source'];h=r['source_sha256'] if field=='source' else r[field+'_sha256'];assert state['frozen_sources'][p]==h==sha(p)
 assert r['original_BF16_retained'] and r['no_new_weight_layout_or_payload'] and r['no_global_tensor_workspace'] and r['all_scratch_call_local'] and r['FP32_output_after_explicit_BF16_round'] and r['compiled']==selection['expected_compiled'];proof=next(v for v in read(EXP/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json') if v['prefix']=='lm_head');bank=r['bank'];assert all(bank[k]==proof[k] for k in ('original_BF16_sha256','packed_sha256','scales_sha256')) and bank['packed_shape']==[130560,2048] and bank['packed_stride']==[2048,1];context=bank['inside_DeviceMemoryProfiler'];assert context['active'] and state['frozen_sources'][context['source']]==context['sha256'];assert len(r['warmups'])==1 and r['warmups'][0]['zero_output'] and r['warmups'][0]['RNG_unchanged'] and r['warmups'][0]['FP32_output_bytes']==33423360
 assert r['profile_probes']
 for p in r['profile_probes']:assert p['explicit_zero_probe_not_real_request'] and p['M']==64 and p['FP32_output_bytes']==33423360 and p['zero_output'] and p['RNG_unchanged'] and p['context']['active'] and not p['context']['manual_KV_bytes'] and state['frozen_sources'][p['context']['source']]==p['context']['sha256']
 for route in r['routes'].values():assert selection['minimum_M']<=route['M']<=64 and route['physical_M']==64 and route['calls']>0 and route['output_dtype']=='float32' and route['kernel_sha256']==r['kernel_sha256'] and route['source_sha256']==r['source_sha256']
 runtime=[v for v in r['routes'].values() if v['phase']=='runtime' and not v['capturing']];actual_M=sorted({v['M'] for v in runtime});return dict(lm_pad64_record=r,new_LM_actual_runtime_M=actual_M,new_LM_runtime_calls=sum(v['calls'] for v in runtime),new_LM_profile_peak_accounted=True,new_LM_original_pack_same=True,new_LM_CUBIN_exact=True,LM_precision_and_route_unchanged=False,legacy_LM64_guard_explicitly_replaced_by_new_integer_counter=True,all_other_legacy_runtime_checks_retained=True)

def audit_startup(root,state):
 r=read(next((root/'lm_pad64_records').glob('route_*.json')));assert r['completed'] and r['compiled']==read(BASE/'selected_plan.json')['expected_compiled'] and len(r['warmups'])==1 and r['profile_probes'];return dict(new_LM_loader_warm_and_explicit_profile_zero64_verified=True,new_LM_startup_routes={k:dict(v) for k,v in r['routes'].items()})
