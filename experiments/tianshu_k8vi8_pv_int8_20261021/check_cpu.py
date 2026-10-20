# SPDX-License-Identifier: Apache-2.0
"""Source reversibility and independent IEEE FP32 RNE / INT64 PV oracle only."""
import ast,hashlib,json,struct
from pathlib import Path
import source
BASE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def main():
    import torch
    torch.set_num_threads(8);rng_before=torch.get_rng_state().clone()
    old_cpu=json.loads((source.PARENT/'CPU_GATE_32g.json').read_text());old_gpu=json.loads((source.PARENT/'GPU_GATE_32g.json').read_text())
    assert old_cpu['complete'] and old_gpu['complete']
    for path,digest in old_cpu['sources'].items():assert sha(path)==digest
    text,proof=source.render(source.SOURCE.read_bytes());ast.parse(text)
    cases=[];checked_quant=checked_integer=0
    for index,(tokens,factor) in enumerate(((32,1.),(64,1.),(128,1.),(32,0.),(64,1.e-12),(128,.25))):
        generator=torch.Generator(device='cpu').manual_seed(202610210900+index);m,d=16,128
        p=torch.rand((m,tokens),generator=generator,dtype=torch.float32)*factor;p[-1].zero_()
        v=torch.randint(-127,128,(tokens,d),generator=generator,dtype=torch.int64).to(torch.int8)
        vs=torch.linspace(.0001,.03,tokens,dtype=torch.float32)
        weighted=p*vs;scale=weighted.abs().amax(1).clamp_min(1.e-30)/127.
        q=(weighted/scale[:,None]).round().clamp(0,127).to(torch.int8)
        independent_q=[];independent_scale=[]
        for row in p.tolist():
            values=[f32(a*b) for a,b in zip(row,vs.tolist())]
            row_scale=f32(max(max(map(abs,values)),f32(1.e-30))/127.)
            independent_scale.append(row_scale);independent_q.append([min(127,max(0,round(f32(a/row_scale)))) for a in values])
        assert torch.equal(q,torch.tensor(independent_q,dtype=torch.int8)) and torch.equal(scale,torch.tensor(independent_scale,dtype=torch.float32))
        integer=q.long()@v.long();reference=[];vrows=v.tolist()
        for row in independent_q:reference.append([sum(row[t]*vrows[t][column] for t in range(tokens)) for column in range(d)])
        assert torch.equal(integer,torch.tensor(reference,dtype=torch.int64))
        assert bool(integer.abs().max()<=tokens*127*127) and tokens*127*127<2**31
        ideal=weighted.double()@v.double();approx=integer.double()*scale.double()[:,None]
        bound=.501*scale.double()[:,None]*v.abs().double().sum(0)[None,:]
        assert bool(((ideal-approx).abs()<=bound+1.e-30).all()) and bool((q[-1]==0).all())
        checked_quant+=q.numel();checked_integer+=integer.numel()
        cases.append(dict(T=tokens,M=m,D=d,factor=factor,exact_RNE_and_INT64=True,max_abs_error=float((ideal-approx).abs().max()),analytical_error_bound_passed=True))
    assert torch.equal(torch.get_rng_state(),rng_before)
    for path in BASE.glob('*.py'):ast.parse(path.read_text())
    paths=[*BASE.glob('*.py'),source.PARENT/'source.py',source.PARENT/'writer.py',source.PARENT/'layout.py',source.PARENT/'reference_quantize.py',source.PARENT/'CPU_GATE_32g.json',source.PARENT/'GPU_GATE_32g.json',source.SOURCE]
    result=dict(complete=True,CPU_only=True,no_GPU_started=True,GPU_not_compiled_or_timed=True,adopted=False,new_Math_not_run=True,
        source_proof=proof,primitive_cases=cases,independent_RNE_elements=checked_quant,independent_INT64_outputs=checked_integer,
        CPU_global_RNG_unchanged=True,integer_maximum_tile_bound=128*127*127,
        scope='CPU oracle and source transformation only; no claim of GPU correctness, speed or model quality',
        sources={str(p):sha(p) for p in paths})
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_proof','sources','primitive_cases')}))
if __name__=='__main__':main()
