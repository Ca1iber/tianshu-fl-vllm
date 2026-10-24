# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
assert not ROOT.exists();CPU.mkdir(parents=True)
s=importlib.util.spec_from_file_location('_Down_small_actual_r4_profile',EXP/'tianshu_W4_LMpad_service_v2_20261023/candidate_profile.py');model=importlib.util.module_from_spec(s);s.loader.exec_module(model)
profile_root=CPU/'r4_profile';profile_root.mkdir();command,env,profile,sources=model.prepare(profile_root,'candidate');assert len(sources)==944
(CPU/'profile.json').write_text(json.dumps(dict(command=command,environment=profile,additional_environment={'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']},sources=sources),indent=2)+'\n')
legacy=json.loads((PARENT/'CPU_GATE_32g.json').read_text());proof=json.loads((PARENT/'INDEPENDENT_CPU_GATE_32g.json').read_text());assert legacy['complete'] and proof['complete'] and proof['all_numeric_passed'] and len(legacy['banks'])==42
assert sha(MODEL)==legacy['checkpoint_sha256']=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
files={str(CPU/'profile.json'):sha(CPU/'profile.json')};banks=legacy['banks'];references=legacy['references']
for entry in banks:
 assert sha(entry['file'])==entry['file_sha256'];files[entry['file']]=entry['file_sha256']
for entry in references:
 assert sha(entry['file'])==entry['sha256'];files[entry['file']]=entry['sha256']
 a=np.load(entry['file'],allow_pickle=False);assert a['H'].shape==(64,6144) and a['original'].shape==a['quantized'].shape==(64,2048);assert np.array_equal(a['H'],hwords(entry['phase'],entry['layer']))
for phase in PHASES:
 path=old.FIXTURE/f'M64_phase{phase}.npz';files[str(path)]=sha(path);assert files[str(path)]==legacy['files'][str(path)]
paths=[*BASE.glob('*.py'),BASE/'PLAN_32g.md',PARENT/'CPU_GATE_32g.json',PARENT/'INDEPENDENT_CPU_GATE_32g.json',PARENT/'common.py',PARENT/'prepare_cpu.py',PARENT/'verify_cpu.py',EXP/'tianshu_dotsaddz_Down64_service_v2_20261023/policy.py',EXP/'tianshu_native_int8_down_service_v2_20261022/native_down_policy.py']
for p,h in legacy['sources'].items():
 if Path(p).name in ('libtorch_cuda.so','libcuinfer.so.7','libcuinfer_blas.so'):paths.append(Path(p))
for p in paths:
 if p.suffix=='.py':ast.parse(p.read_text())
 sources[str(p.resolve())]=sha(p)
assert all(sha(p)==h for p,h in sources.items())
for p,h in sources.items():
 target=CPU/'source_snapshot'/Path(p).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
 if Path(p).suffix in ('.py','.json','.md','.sha256'):target.write_bytes(Path(p).read_bytes());assert sha(target)==h
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
gate=dict(complete=True,CPU_only=True,CUDA_initialized=False,first_GPU_not_run=True,sources=sources,files=files,banks=banks,references=references,checkpoint_sha256=legacy['checkpoint_sha256'],original_model_source_count=944,original_pack_and_FP64_reference_reused=True,no_new_large_weight_arrays=True,Ms=MS,numeric_layers=LAYERS,model_quality_qualified=False,adopted=False)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(dict(complete=True,CPU_only=True,sources=len(sources),files=len(files),banks=len(banks),references=len(references))),flush=True)
