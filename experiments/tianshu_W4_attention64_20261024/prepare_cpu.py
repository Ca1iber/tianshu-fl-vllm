# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
assert not ROOT.exists();CPU.mkdir(parents=True)
s=importlib.util.spec_from_file_location('_attention64_current_r4',EXP/'tianshu_W4_LMpad_service_v2_20261023/candidate_profile.py');model=importlib.util.module_from_spec(s);s.loader.exec_module(model);pr=CPU/'r4_profile';pr.mkdir();command,env,profile,sources=model.prepare(pr,'candidate');assert len(sources)==944
(CPU/'profile.json').write_text(json.dumps(dict(command=command,environment=profile,additional_environment={'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']}),indent=2)+'\n')
oldroot=Path('/workspace/benchmark-results/20261022_native_int8_attention64_GPU_tianshu');previous=json.loads((oldroot/'result.json').read_text());assert previous['complete'] and len(previous['banks'])==84;old_banks={(b['layer'],b['role']):b for b in previous['banks']}
route=Path('/workspace/benchmark-results/20261023_W4_LMpad_v3_official4k_tianshu/native_attention_records/route_93241.json');actual=json.loads(route.read_text());assert actual['completed'] and actual['rows_by_role']=={'QKV':[128,512,4096],'O':[512,4096]}
for role in ROLES:assert len({r['bank_id'] for r in actual['routes'].values() if r['role']==role and r['M']==64 and r['capturing'] and r['implementation']=='qualified_BF16_vendor_mm'})==42
files={str(CPU/'profile.json'):sha(CPU/'profile.json'),str(oldroot/'result.json'):sha(oldroot/'result.json'),str(route):sha(route)};banks=[];refs=[]
assert sha(MODEL)=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
for layer in range(42):
 for role in ROLES:
  w=original(layer,role);assert ahash(w)==old_banks[(layer,role)]['original_BF16_sha256'];bank=dict(layer=layer,role=role,N=w.shape[0],K=2048,original_sha256=ahash(w),variants={})
  xs={phase:inputs(layer,role,phase) for phase in PHASES}
  for phase,x in xs.items():p=CPU/f'{role}_L{layer}_X{phase}.npy';np.save(p,x,allow_pickle=False);files[str(p)]=sha(p)
  targets={}
  if layer in LAYERS:
   original_fp64=f32(w).T.astype(np.float64)
   targets={phase:{'original':f32(x).astype(np.float64)@original_fp64} for phase,x in xs.items()}
  for mode in MODES:
   q,s,z,p=quant(w,mode);assert np.array_equal(unpack(p),q) and np.isfinite(f32(s)).all() and np.isfinite(f32(z)).all();path=CPU/f'{role}_L{layer}_{mode}.npz';np.savez(path,W=p,S=s,Z=z);files[str(path)]=sha(path);bank['variants'][mode]=dict(file=str(path),file_sha256=files[str(path)],packed_sha256=ahash(p),scale_sha256=ahash(s),zero_sha256=ahash(z),payload_bytes=int(p.nbytes+s.nbytes+z.nbytes))
   if layer in LAYERS:
    dq=f32(dequant(q,s,z)).astype(np.float64)
    for phase,x in xs.items():targets[phase][mode]=f32(x).astype(np.float64)@dq
  if layer in LAYERS:
   for phase,data in targets.items():path=CPU/f'{role}_L{layer}_reference{phase}.npz';np.savez(path,**data);files[str(path)]=sha(path);refs.append(dict(layer=layer,role=role,phase=phase,file=str(path),sha256=files[str(path)]))
  banks.append(bank)
 print(json.dumps(dict(layer=layer,packed_both_roles=True,full_FP64_reference=layer in LAYERS)),flush=True)
paths=[*BASE.glob('*.py'),BASE/'PLAN_32g.md',Path(actual['source']),EXP/'tianshu_W4_affine_error_CPU_20261023/CPU_GATE_32g.json']
old_cpu=json.loads((EXP/'tianshu_native_dotsaddz_down42_20261023/CPU_GATE_32g.json').read_text())
for p,h in old_cpu['sources'].items():
 if Path(p).name in ('libtorch_cuda.so','libcuinfer.so.7','libcuinfer_blas.so'):paths.append(Path(p))
for p in paths:
 if p.suffix=='.py':ast.parse(p.read_text())
 sources[str(p.resolve())]=sha(p)
assert all(sha(p)==h for p,h in sources.items()) and sha(MODEL)=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
for p,h in sources.items():
 if Path(p).suffix in ('.py','.json','.md','.sha256'):
  dest=CPU/'source_snapshot'/Path(p).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(p).read_bytes());assert sha(dest)==h
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
gate=dict(complete=True,CPU_only=True,first_GPU_not_run=True,checkpoint_sha256=sha(MODEL),sources=sources,files=files,banks=banks,references=refs,original_r4_source_count=944,all84_original_weights_SHA_match=True,all168_full_pack_roundtrips_passed=True,actual_M64_BF16_capture_roles_verified=True,model_quality_qualified=False,adopted=False)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(dict(complete=True,banks=len(banks),references=len(refs),files=len(files),sources=len(sources))),flush=True)
