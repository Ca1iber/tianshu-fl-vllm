# SPDX-License-Identifier: Apache-2.0
"""Independent FP32 operation oracle and exact source-difference checks."""
import ast,hashlib,json,math,random,struct
from pathlib import Path
import source
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
def main():
    for path in BASE.glob('*.py'):ast.parse(path.read_text())
    gate=json.loads((source.PARENT/'CPU_GATE_32g.json').read_text());assert gate['complete']
    for path,digest in gate['sources'].items():assert sha(path)==digest
    original=(source.PARENT/'kernel_pv.py').read_text()
    edits=[('def quantize_probabilities(probability,value_scales):','def quantize_probabilities(probability,value_scales,RECIPROCAL:tl.constexpr=False):'),
        ('def quantized_pv(probability,values,value_scales):','def quantized_pv(probability,values,value_scales,RECIPROCAL:tl.constexpr=False):'),
        ("\n    quantized,scale=quantize_probabilities(probability,value_scales)\n","\n    quantized,scale=quantize_probabilities(probability,value_scales,RECIPROCAL)\n"),
        ('    quantized=libdevice.nearbyint(tl.div_rn(weighted,scale[:,None]))','    if RECIPROCAL:\n        inverse=tl.div_rn(1.,scale)\n        quantized=libdevice.nearbyint(weighted*inverse[:,None])\n    else:\n        quantized=libdevice.nearbyint(tl.div_rn(weighted,scale[:,None]))')]
    transformed=original
    for before,after in edits:assert transformed.count(before)==1;transformed=transformed.replace(before,after)
    assert transformed==(BASE/'kernel_pv.py').read_text()
    text,proof=source.render(source.SOURCE.read_bytes());ast.parse(text)
    generator=random.Random(202610210332);cases=[];different=total=0
    for tokens,factor in ((32,1.),(64,1.),(32,0.),(32,1.e-12),(64,.25)):
        max_difference=0;case_diff=0
        for row in range(16):
            weighted=[f32(f32(generator.random()*factor)*f32(.0001+generator.random())) for _ in range(tokens)]
            peak=max(max(map(abs,weighted)),f32(1.e-30));scale=f32(peak/127.);inverse=f32(1./scale)
            assert math.isfinite(inverse)
            strict=[min(127,max(0,round(f32(v/scale)))) for v in weighted]
            fast=[min(127,max(0,round(f32(v*inverse)))) for v in weighted]
            assert all(abs(a-b)<=1 for a,b in zip(strict,fast))
            assert all(abs(v-q*scale)<=.5001*scale+1.e-38 for v,q in zip(weighted,fast))
            case_diff+=sum(a!=b for a,b in zip(strict,fast));total+=tokens
            max_difference=max(max_difference,max(abs(a-b) for a,b in zip(strict,fast)))
        different+=case_diff;cases.append(dict(tokens=tokens,factor=factor,max_integer_difference=max_difference,changed_quantized_values=case_diff))
    probe=(BASE/'probe_gpu.py').read_text();assert "kw['PV_RECIPROCAL']=self.mode=='PVR32'" in probe
    verifier=(BASE/'verify_gpu.py').read_text();assert "record['PV_RECIPROCAL']==(m=='PVR32')" in verifier
    paths=set(gate['sources'])|{str(p) for p in [*BASE.glob('*.py'),*BASE.glob('*.sh'),source.PARENT/'CPU_GATE_32g.json']}
    result=dict(complete=True,CPU_only=True,no_GPU_started=True,adopted=False,new_Math_not_run=True,
        parent_primitive_GPU_evidence_does_not_prove_new_reciprocal_rounding=True,new_GPU_numeric_checks_required=True,
        exact_helper_source_transformation_checked=True,independent_CPU_FP32_values=total,changed_quantized_values=different,cases=cases,
        comparison_modes=['K8V16','K8VI8Heads','PV32','PVR32'],strict_vs_reciprocal_same_tile_and_warps=True,
        source_proof=proof,sources={p:sha(p) for p in sorted(paths)})
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_proof','sources')}))
if __name__=='__main__':main()
