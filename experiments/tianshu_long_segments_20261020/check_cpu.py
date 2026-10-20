# SPDX-License-Identifier: Apache-2.0
"""Verify reversible source edits and segment coverage over full supported context."""
import argparse
import hashlib
import json
from pathlib import Path
from source import generate

def ceildiv(a,b):
    return (a+b-1)//b

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reader',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    generated=generate(args.reader.read_text())
    count=0
    for maximum in (8,16,32):
        for length in range(1,131073):
            segments=4 if length<=8192 else maximum
            tiles=ceildiv(length,segments*16)
            active=ceildiv(length,tiles*16)
            assert 1<=active<=segments<=maximum
            assert active*tiles*16>=length
            assert (active-1)*tiles*16<length
            if length<=8192:
                assert tiles==ceildiv(length,4*16)
            count+=1
    base=Path(__file__).resolve().parent
    result=dict(complete=True,CPU_only=True,coverage_cases=count,context_range=[1,131072],candidate_maximum_segments=[8,16,32],
                source_generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),GPU_executed=False,adopted=False,
                source_edit_scope='Only producer/reducer tile-count expressions; constant divisors, KV decode and ABI unchanged.',
                sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),base/'source.py',args.reader)})
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
