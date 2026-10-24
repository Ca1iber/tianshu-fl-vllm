# SPDX-License-Identifier: Apache-2.0
"""Freeze exactly the current quantized weights; only physical format/backend changes."""
import ast,sys
from common import *
assert not ROOT.exists();CPU.mkdir(parents=True)
old=json.loads((PARENT/'CPU_GATE_32g.json').read_text())
proof=json.loads((PARENT/'INDEPENDENT_CPU_GATE_32g.json').read_text())
formats=json.loads((FORMAT/'FUNCTIONAL_GATE_32g.json').read_text())
assert old['complete'] and proof['complete'] and proof['all_numeric_passed']
assert formats['complete'] and 'generic_high_even' in formats['passed_variants']
payloads={path:h for h,path in (line.split(None,1) for line in RELEASE_PAYLOADS.read_text().splitlines())}
banks=[];files={}
for bank in old['banks']:
    path=bank['file'];assert sha(path)==bank['file_sha256']==payloads[path]
    data=np.load(path,allow_pickle=False)
    raw,params,format_proof=generic_arrays(data['W'],data['S'],data['Z'])
    banks.append(dict(**bank,generic=format_proof))
    files[path]=bank['file_sha256']
references=old['references'];assert len(references)==9
for ref in references:
    path=ref['file'];assert sha(path)==ref['sha256'];files[path]=ref['sha256']
    data=np.load(path,allow_pickle=False)
    assert data['quantized'].shape==(64,2048) and np.array_equal(data['H'],parent.hwords(ref['phase'],ref['layer']))
for phase in PHASES:
    path=parent.FIXTURE/f'M64_phase{phase}.npz';assert sha(path)==old['files'][str(path)];files[str(path)]=sha(path)
paths=set(BASE.glob('*.py'))|{BASE/'PLAN_32g.md',PARENT/'common.py',PARENT/'CPU_GATE_32g.json',
    PARENT/'INDEPENDENT_CPU_GATE_32g.json',FORMAT/'FUNCTIONAL_GATE_32g.json',FORMAT/'prepare_cpu.py',
    FORMAT/'probe_gpu.py',RELEASE_PAYLOADS}
sources={str(path):sha(path) for path in paths}
libraries={p:h for p,h in old['sources'].items() if Path(p).name in ('libtorch_cuda.so','libcuinfer.so.7','libcuinfer_blas.so')}
assert len(libraries)==3 and all(sha(p)==h for p,h in libraries.items())
assert sha(parent.MODEL)==old['checkpoint_sha256']
for path,h in sources.items():
    if Path(path).suffix=='.py':ast.parse(Path(path).read_text())
    target=CPU/'source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(Path(path).read_bytes());assert sha(target)==h
assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
result=dict(complete=True,CPU_only=True,banks=banks,references=references,sources=sources,libraries=libraries,files=files,
    current_r5_all42_payloads_exact=True,all42_same_full_dequantized_BF16_words=True,
    high_even_physical_layout_roundtrip_passed=True,shape=dict(M=64,K=6144,N=2048,group=128,inner_k_tiles=8),
    native_converter_only_startup=True,checkpoint_sha256=old['checkpoint_sha256'],
    variants=VARIANTS,numeric_layers=LAYERS,numeric_phases=PHASES,model_or_service_qualified=False,adopted=False)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('banks','references','sources','libraries','files')},indent=2))
