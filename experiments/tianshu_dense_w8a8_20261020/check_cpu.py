# SPDX-License-Identifier: Apache-2.0
"""Independent round-to-even examples and signed integer accumulation bounds."""
from fractions import Fraction
import hashlib,json,math,struct
from pathlib import Path
BASE=Path(__file__).resolve().parent

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def round_even(x):
    exact=Fraction(x);floor=exact.numerator//exact.denominator
    remainder=exact-floor
    return floor+int(remainder>Fraction(1,2) or (remainder==Fraction(1,2) and floor%2))

def main():
    cases=0
    for maximum in (0.,1.e-12,.25,1.,3.,127.):
        scale=f32(max(maximum,1.e-8)/127.)
        for i in range(-254,255):
            value=f32(i*maximum/254)
            divided=f32(value/scale)
            packed=max(-127,min(127,round_even(divided)))
            assert -127<=packed<=127
            assert packed==max(-127,min(127,round(divided)))
            cases+=1
    ties=[(i+.5,round_even(i+.5)) for i in range(-127,127)]
    assert all(answer%2==0 for _,answer in ties)
    assert max(259,2048,6144)*127*127<2**31
    sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),BASE/'kernel.py')}
    gate=dict(complete=True,CPU_only=True,GPU_not_executed=True,rounding_cases=cases,halfway_tie_cases=len(ties),
        largest_inner_dimension=6144,worst_case_accumulator_abs=6144*127*127,INT32_no_overflow=True,
        no_model_accuracy_or_performance_claim=True,quantized_range=[-127,127],sources=sources)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
