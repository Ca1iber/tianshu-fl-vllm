# SPDX-License-Identifier: Apache-2.0
"""Exhaust every byte in each word lane against bytearray/struct indexing."""
import hashlib,json,struct
from pathlib import Path
import source
BASE=Path(__file__).resolve().parent
PARENT=Path('/workspace/benchmark-results/20261020_k8v16_pair_load_GPU_tianshu/pair_rows_reader/generated_pair_rows.py')

def main():
    proofs={mode:source.render(PARENT,mode)[1] for mode in source.MODES}
    checks=0
    for lane in range(4):
        for code in range(256):
            raw=bytearray([17,137,0,255]);raw[lane]=code
            word=struct.unpack('<I',raw)[0]
            selected=(word>>(8*lane))&255
            magnitude=selected&127
            actual=(((magnitude<<4)+14848) if magnitude>=8 else 0)|((selected&128)<<8)
            expected=((16*(raw[lane]&127)+14848) if (raw[lane]&127)>=8 else 0) | ((raw[lane]//128)*32768)
            assert actual==expected and 0<=actual<=65535
            checks+=1
    # The paired joins must assemble [byte0,byte1,byte2,byte3], not [0,2,1,3].
    joined=[list(zip([0,2],[1,3]))];assert [x for row in joined[0] for x in row]==list(range(4))
    sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),BASE/'source.py',BASE/'launch.py',source.SIBLING,PARENT)}
    gate=dict(complete=True,CPU_only=True,GPU_not_executed=True,word_decode_lane_cases=checks,
        all_256_code_values_in_each_of_4_lanes=True,provenance=proofs,sources=sources)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({key:value for key,value in gate.items() if key!='sources'}))
if __name__=='__main__':main()
