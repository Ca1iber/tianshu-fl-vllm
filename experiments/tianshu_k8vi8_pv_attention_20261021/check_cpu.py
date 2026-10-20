# SPDX-License-Identifier: Apache-2.0
"""Freeze source/primitive proofs and verify all changed probe launch choices."""
import ast,hashlib,json
from pathlib import Path
import source
BASE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parent_gate=json.loads((source.PARENT/'CPU_GATE_32g.json').read_text())
    primitive=json.loads((source.PARENT/'PRIMITIVE_GPU_GATE_32g.json').read_text())
    assert parent_gate['complete'] and primitive['complete']
    for path,digest in parent_gate['sources'].items():assert sha(path)==digest
    rendered,proof=source.render(source.SOURCE.read_bytes());ast.parse(rendered)
    probe=(BASE/'probe_gpu.py').read_text();ast.parse(probe)
    assert "cases=((1,33),(8,257),(40,4096),(64,4096),(64,5120))" in probe
    assert "kw['PV_INT8']=self.mode.startswith('PV')" in probe
    assert "TILE_SIZE=int(self.mode[2:])" in probe and "num_warps=2 if self.mode.startswith('PV') else 1" in probe
    assert "CPU_FP64_reference=True" in probe and "reader,helper,proof=source.load_modules" in probe
    assert "writer_{f['mode']}_case{case_id}_phase{phase}.pt" in probe
    for name in ('writer.py','reference_quantize.py'):assert (BASE/name).read_bytes()==(source.PARENT.parent/'tianshu_k8vi8_heads_v2_20261021'/name).read_bytes()
    paths=set(parent_gate['sources'])|{str(p) for p in [*BASE.glob('*.py'),*BASE.glob('*.sh'),source.PARENT/'CPU_GATE_32g.json',source.PARENT/'PRIMITIVE_GPU_GATE_32g.json',source.PARENT/'PRIMITIVE_CASES_32g.json']}
    result=dict(complete=True,CPU_only=True,no_GPU_started=True,adopted=False,Math_not_run=True,full_attention_not_yet_compiled=True,
        primitive_GPU_verification_checked=True,real_native_page_layout=True,writer_timed=True,three_changed_input_phases=True,
        configurations={'K8V16':dict(tile=16,warps=1,integer_PV=False),'K8VI8Heads':dict(tile=16,warps=1,integer_PV=False),'PV32':dict(tile=32,warps=2,integer_PV=True),'PV64':dict(tile=64,warps=2,integer_PV=True)},
        source_proof=proof,sources={p:sha(p) for p in sorted(paths)})
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','source_proof')}))
if __name__=='__main__':main()
