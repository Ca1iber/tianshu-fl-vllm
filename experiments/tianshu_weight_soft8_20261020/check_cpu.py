# SPDX-License-Identifier: Apache-2.0
"""Exhaustive codec contract only; no GPU or model-quality claim."""
import argparse
import hashlib
import json
from pathlib import Path
import codec

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    counts={}
    for base in codec.BASES:
        for bits in range(65536):
            actual=codec.decode_bits(codec.encode_bits(bits,base),base)
            expected=codec.independent_reference(bits,base)
            assert actual==expected,(base,bits,actual,expected)
        for code in range(256):
            bits=codec.decode_bits(code,base)
            assert bits<65536
            assert codec.encode_bits(bits,base)==(code if code&127>=8 else code&128)
        counts[str(base)]=dict(exhaustive_BF16_patterns=65536,decoded_codes=256)
    base=Path(__file__).resolve().parent
    paths=[base/name for name in ('codec.py','kernel.py','check_cpu.py')]
    result=dict(complete=True,CPU_only=True,exhaustive_patterns=65536*len(codec.BASES),cases=counts,
                encoding_matches_independent_real_number_rounding=True,checkpoint_files_never_modified=True,
                GPU_executed=False,model_executed=False,Math_not_run=True,adopted=False,
                sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
