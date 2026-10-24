# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
assert not ROOT.exists();CPU.mkdir(parents=True)
gate=json.loads((PARENT/'CPU_GATE_32g.json').read_text())
proof=json.loads((PARENT/'INDEPENDENT_CPU_GATE_32g.json').read_text())
cap=json.loads((CAPABILITY/'GPU_CPU_GATE_32g.json').read_text())
assert gate['complete'] and proof['complete'] and proof['all_numeric_passed'] and cap['complete'] and cap['FP16_tested_contract_accepted']
payloads={path:sha256 for sha256,path in (line.split(None,1) for line in RELEASE_PAYLOADS.read_text().splitlines())}
files={};banks=[];references=[];exact_scale=exact_zero=0
oldrefs={(r['layer'],r['phase']):r for r in gate['references']}
for bank in gate['banks']:
    layer=bank['layer'];path=bank['file']
    assert sha(path)==bank['file_sha256']==payloads[path];files[path]=bank['file_sha256']
    a=np.load(path,allow_pickle=False);s,z=f32(a['S']),f32(a['Z'])
    sh,zh=s.astype(np.float16),z.astype(np.float16)
    assert np.isfinite(sh).all() and np.isfinite(zh).all()
    exact_scale+=int(np.array_equal(sh.astype(np.float32),s))
    exact_zero+=int(np.array_equal(zh.astype(np.float32),z))
    banks.append(bank)
    if layer not in LAYERS:continue
    q=parent.unpack(a['W'])
    deq=(q.astype(np.float32)*np.repeat(sh.astype(np.float32),128,axis=0)+np.repeat(zh.astype(np.float32),128,axis=0)).astype(np.float16).astype(np.float64)
    for phase in PHASES:
        original=oldrefs[(layer,phase)];assert sha(original['file'])==original['sha256']
        files[original['file']]=original['sha256']
        old=np.load(original['file'],allow_pickle=False);h=parent.hwords(phase,layer)
        assert np.array_equal(h,old['H'])
        x=f32(h);xh=x.astype(np.float16);assert np.isfinite(xh).all()
        target=CPU/f'reference_L{layer}_phase{phase}.npz'
        np.savez(target,H=h,H_half_words=xh.view(np.uint16),BF16=old['quantized'],
            FP16=xh.astype(np.float64)@deq,original=old['original'])
        files[str(target)]=sha(target)
        references.append(dict(layer=layer,phase=phase,file=str(target),sha256=files[str(target)],
            activation_changed_cells=int(np.count_nonzero(xh.astype(np.float32)!=x)),
            activation_max_conversion_abs=float(abs(xh.astype(np.float32)-x).max())))
    print(json.dumps(dict(layer=layer,FP64_phases=3)),flush=True)
for phase in PHASES:
    path=parent.FIXTURE/f'M64_phase{phase}.npz';assert sha(path)==gate['files'][str(path)];files[str(path)]=sha(path)
paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',PARENT/'common.py',PARENT/'CPU_GATE_32g.json',PARENT/'INDEPENDENT_CPU_GATE_32g.json',
    CAPABILITY/'GPU_CPU_GATE_32g.json',CAPABILITY/'CPU_GATE_32g.json',RELEASE_PAYLOADS}
sources={str(path):sha(path) for path in paths}
libraries={p:h for p,h in gate['sources'].items() if Path(p).name in ('libtorch_cuda.so','libcuinfer.so.7','libcuinfer_blas.so')}
assert len(libraries)==3 and all(sha(p)==h for p,h in libraries.items())
assert sha(parent.MODEL)==gate['checkpoint_sha256']
for path,h in sources.items():
    if Path(path).suffix=='.py':ast.parse(Path(path).read_text())
    target=CPU/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(Path(path).read_bytes());assert sha(target)==h
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
result=dict(complete=True,CPU_only=True,banks=banks,references=references,sources=sources,libraries=libraries,files=files,
    current_r5_Down_all42_payloads_exact=True,scale_conversion_exact_banks=exact_scale,zero_conversion_exact_banks=exact_zero,
    shape=dict(M=64,K=6144,N=2048,group=128),checkpoint_sha256=gate['checkpoint_sha256'],
    numerical_full_FP64_layers=LAYERS,variants=VARIANTS,model_or_service_qualified=False,adopted=False)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('banks','references','sources','libraries','files')},indent=2))
