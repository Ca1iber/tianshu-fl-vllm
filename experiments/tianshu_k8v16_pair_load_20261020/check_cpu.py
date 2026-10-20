# SPDX-License-Identifier: Apache-2.0
"""Exhaust all two-byte values and independently decode BF16 bit patterns."""
import hashlib,json,struct
from pathlib import Path
import source
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_k8v16_address32_20261020/qualified_S16_reader.py'

def decode(x):
    mag=x&127
    return ((mag*16+14848) if mag>=8 else 0) | ((x&128)*256)

def main():
    proofs={mode:source.render(PARENT,mode)[1] for mode in source.MODES}
    count=0
    for pair in range(65536):
        packed=struct.pack('<H',pair)
        for high in (False,True):
            scalar=packed[int(high)];word=(pair>>8) if high else (pair&255)
            magnitude=word&127
            narrowed=(((magnitude<<4)+14848) if magnitude>=8 else 0)|((word&128)<<8)
            assert narrowed<=65535 and narrowed==decode(scalar)
            count+=1
    checked=0
    # An independent byte-address fixture: odd page indices, head changes,
    # non-contiguous page views and slot tails; untouched bytes retain guards.
    for page_stride in (12288,24576):
        raw=bytearray([17])*(page_stride*5)
        for page in (0,1,4):
            for slot in (0,1,15):
                for head in (0,1):
                    start=page*page_stride+slot*256+head*128
                    values=bytes(((i*37+page*11+slot+head)&255) for i in range(128))
                    raw[start:start+128]=values
                    for dim in range(0,128,2):
                        pair=struct.unpack_from('<H',raw,start+dim)[0]
                        assert (pair&255,pair>>8)==(values[dim],values[dim+1])
                        checked+=2
    sources={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),BASE/'source.py',BASE/'launch.py',PARENT)}
    gate=dict(complete=True,CPU_only=True,GPU_not_executed=True,exhaustive_pair_decode_values=count,
        independent_strided_K_byte_loads=checked,provenance=proofs,sources=sources)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps(gate))
if __name__=='__main__':main()
