# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
assert not ROOT.exists();CPU.mkdir(parents=True)
spec=importlib.util.spec_from_file_location('_fitO_component_r4',EXP/'tianshu_W4_LMpad_service_v2_20261023/candidate_profile.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
profile_root=CPU/'r4_profile';profile_root.mkdir()
command,env,profile,sources=model.prepare(profile_root,'candidate');assert len(sources)==944
(CPU/'profile.json').write_text(json.dumps(dict(command=command,environment=profile,additional_environment={'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']}),indent=2)+'\n')
fit_folder=EXP/'tianshu_W4_QKV_affine_fit_CPU_20261024'
fit_proof=json.loads((fit_folder/'INDEPENDENT_CPU_GATE_32g.json').read_text())
fit_root=Path('/dev/shm/tianshu_W4_QKV_affine_fit_CPU_20261024')
assert fit_proof['complete'] and fit_proof['all_groups_actual_BF16_MSE_nonincreasing']
assert sha(fit_root/'result.json')==fit_proof['raw_result_sha256']
fit=json.loads((fit_root/'result.json').read_text());fit_banks={b['layer']:b for b in fit['banks']}
old=json.loads((EXP/'tianshu_W4_attention64_20261024/CPU_GATE_32g.json').read_text())
old_banks={b['layer']:b for b in old['banks'] if b['role']=='QKV'}
files={str(CPU/'profile.json'):sha(CPU/'profile.json'),str(fit_root/'result.json'):sha(fit_root/'result.json')}
files.update(fit['files'])
banks=[];references=[]
assert sha(MODEL)=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
for layer in range(42):
    w=original(layer,'QKV');assert ahash(w)==old_banks[layer]['original_sha256']==fit_banks[layer]['original_sha256']
    bank=dict(layer=layer,role='QKV',N=2560,K=2048,original_sha256=ahash(w),variants={})
    variants={}
    q,s,z,packed=quant(w,'affine')
    assert ahash(packed)==old_banks[layer]['variants']['affine']['packed_sha256']
    assert ahash(s)==old_banks[layer]['variants']['affine']['scale_sha256']
    assert ahash(z)==old_banks[layer]['variants']['affine']['zero_sha256']
    variants['minmax']=(packed,s,z)
    data=np.load(fit_root/f'QKV_L{layer}_fitted.npz',allow_pickle=False)
    variants['fitted']=(data['w'],data['s'],data['z'])
    assert [ahash(a) for a in variants['fitted']]==[fit_banks[layer][key] for key in ('packed_sha256','scale_sha256','zero_sha256')]
    targets={}
    for phase in PHASES:
        x=inputs(layer,'QKV',phase);p=CPU/f'QKV_L{layer}_X{phase}.npy';np.save(p,x,allow_pickle=False);files[str(p)]=sha(p)
        if layer in LAYERS: targets[phase]={'original':f32(x).astype(np.float64)@f32(w).T.astype(np.float64)}
    for mode,(packed,s,z) in variants.items():
        q=unpack(packed);assert np.array_equal(pack(q),packed)
        path=CPU/f'QKV_L{layer}_{mode}.npz';np.savez(path,W=packed,S=s,Z=z);files[str(path)]=sha(path)
        bank['variants'][mode]=dict(file=str(path),file_sha256=files[str(path)],packed_sha256=ahash(packed),scale_sha256=ahash(s),zero_sha256=ahash(z),payload_bytes=packed.nbytes+s.nbytes+z.nbytes)
        if layer in LAYERS:
            dq=f32(dequant(q,s,z)).astype(np.float64)
            for phase in PHASES:targets[phase][mode]=f32(inputs(layer,'QKV',phase)).astype(np.float64)@dq
    for phase,data in targets.items():
        path=CPU/f'QKV_L{layer}_reference{phase}.npz';np.savez(path,**data);files[str(path)]=sha(path)
        references.append(dict(layer=layer,role='QKV',phase=phase,file=str(path),sha256=files[str(path)]))
    banks.append(bank)
    print(json.dumps(dict(layer=layer,full_FP64=layer in LAYERS)),flush=True)
paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md'}|set(fit_folder.glob('*.py'))|{fit_folder/name for name in ('PLAN_32g.md','CPU_GATE_32g.json','INDEPENDENT_CPU_GATE_32g.json')}
paths|={EXP/'tianshu_W4_attention64_20261024'/name for name in ('CPU_GATE_32g.json','INDEPENDENT_CPU_GATE_32g.json','EXECUTION.sha256')}
libgate=json.loads((EXP/'tianshu_native_dotsaddz_down42_20261023/CPU_GATE_32g.json').read_text())
paths|={Path(p) for p in libgate['sources'] if Path(p).name in ('libtorch_cuda.so','libcuinfer.so.7','libcuinfer_blas.so')}
for p in paths:
    if p.suffix=='.py':ast.parse(p.read_text())
    sources[str(p.resolve())]=sha(p)
assert all(sha(p)==h for p,h in sources.items()) and all(sha(p)==h for p,h in files.items())
assert sha(MODEL)=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
for p,h in sources.items():
    if Path(p).suffix in ('.py','.json','.md','.sha256'):
        target=CPU/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(p).read_bytes());assert sha(target)==h
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
gate=dict(complete=True,CPU_only=True,first_GPU_not_run=True,checkpoint_sha256=sha(MODEL),sources=sources,files=files,banks=banks,references=references,
    original_r4_source_count=944,all42_original_weights_SHA_match=True,all84_full_pack_roundtrips_passed=True,model_quality_qualified=False,adopted=False)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps(dict(complete=True,banks=len(banks),references=len(references),sources=len(sources),files=len(files))),flush=True)
