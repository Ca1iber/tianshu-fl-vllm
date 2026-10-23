# SPDX-License-Identifier: Apache-2.0
import ast,sys
from common import *
assert not CPU.exists() and not (BASE/'CPU_GATE_32g.json').exists();CPU.mkdir()
parent=json.loads((PARENT/'CPU_GATE_32g.json').read_text());proof=json.loads((PARENT/'INDEPENDENT_CPU_GATE_32g.json').read_text())
assert parent['complete'] and proof['complete'] and proof['all_numeric_passed'] and sha(Path(proof['raw_directory'])/'result.json')==proof['result_sha256']
sources=dict(parent['sources'])
for p in [*BASE.glob('*.py'),BASE/'PLAN_32g.md',BASE/'run.sh',PARENT/'CPU_GATE_32g.json',PARENT/'INDEPENDENT_CPU_GATE_32g.json',EXP/'tianshu_native_W4_formats_20261023/CONTRACT_STRINGS_32g.txt']:
    if p.suffix=='.py':ast.parse(p.read_text())
    sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items()) and all(sha(p)==h for p,h in parent['files'].items())
files=dict(parent['files']);group_banks={'128':parent['banks']};references=[dict(group=128,**r) for r in parent['references']]
for group in (64,256):
    group_banks[str(group)]=[]
    for layer in range(42):
        w=weight(layer);q,s,p,z=quant(w,group)
        assert np.array_equal(unpack(p),q) and np.array_equal(f32(z),-8*f32(s))
        path=CPU/f'G{group}_layer{layer:02}.npz';np.savez(path,W=p,S=s,Z=z);files[str(path)]=sha(path)
        group_banks[str(group)].append(dict(layer=layer,original_sha256=ahash(w),pack_sha256=ahash(p),scale_sha256=ahash(s),zero_sha256=ahash(z),file=str(path),file_sha256=files[str(path)]))
        if layer in LAYERS:
            deq=bf16((q.astype(np.int16)-8).astype(np.float32)*np.repeat(f32(s),group,axis=0));dq=f32(deq).astype(np.float64)
            for phase in PHASES:
                h=hwords(phase,layer);hf=f32(h).astype(np.float64)
                oldref=next(r for r in parent['references'] if r['layer']==layer and r['phase']==phase)
                with np.load(oldref['file'],allow_pickle=False) as ref:
                    assert np.array_equal(h,ref['H']);original=ref['original'].copy()
                path=CPU/f'G{group}_reference_L{layer}_phase{phase}.npz';np.savez(path,H=h,original=original,quantized=hf@dq);files[str(path)]=sha(path)
                references.append(dict(group=group,layer=layer,phase=phase,file=str(path),sha256=files[str(path)]))
        print(json.dumps(dict(group=group,layer=layer,pack_roundtrip=True,FP64_reference=layer in LAYERS)),flush=True)
assert sha(MODEL)==parent['checkpoint_sha256'] and all(sha(p)==h for p,h in sources.items()) and all(sha(p)==h for p,h in files.items())
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
g=dict(complete=True,CPU_only=True,no_GPU_imports=True,sources=sources,files=files,banks=parent['banks'],group_banks=group_banks,references=references,groups=GROUPS,checkpoint_sha256=parent['checkpoint_sha256'],all_new_84_bank_pack_roundtrips_exact=True,existing_G128_files_reused_SHA_exact=True,first_GPU_not_run=True,adopted=False)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps({k:v for k,v in g.items() if k not in ('sources','files','banks','group_banks','references')}))
