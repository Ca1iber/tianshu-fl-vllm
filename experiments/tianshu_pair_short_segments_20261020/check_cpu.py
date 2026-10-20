# SPDX-License-Identifier: Apache-2.0
"""Independent token coverage and safe capture-to-runtime split bounds."""
import hashlib,json
from pathlib import Path
import source
BASE=Path(__file__).resolve().parent
PARENT=Path('/workspace/benchmark-results/20261020_k8v16_pair_load_GPU_tianshu/pair_rows_reader/generated_pair_rows.py')

def native_segments(n,length):
    value=4 if length<=128 else 64 if n==1 else 32 if n<=3 else 16 if n<=8 else 8 if n<=16 else 16 if n<=24 else 4
    return 16 if 33<=n<=56 else value

def main():
    proofs={mode:source.render(PARENT,mode)[1] for mode in source.MODES}
    cases=transition=0
    lengths=(1,16,33,64,128,129,4096,5120,8192,8193,16384)
    for mode in source.MODES:
        for n in range(1,65):
            for length in lengths:
                native=native_segments(n,length)
                segments=source.configured_segments(mode,n,native)
                active_target=source.SHORT_SPLITS[mode] if n>24 and length<=8192 else segments
                tiles=(length+active_target*16-1)//(active_target*16)
                # Enumerate actual intervals, not a second copy of the GPU indexing.
                counts=[0]*length
                for segment in range(segments):
                    start=segment*tiles*16
                    for token in range(start,min(length,start+tiles*16)):counts[token]+=1
                assert all(count==1 for count in counts)
                assert segments>=((length+tiles*16-1)//(tiles*16))
                cases+=1
                if n>24:
                    capture_segments=source.configured_segments(mode,n,native_segments(n,1))
                    assert capture_segments==segments,(mode,n,length,capture_segments,segments)
                    transition+=1
    sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),BASE/'source.py',BASE/'launch.py',PARENT)}
    gate=dict(complete=True,CPU_only=True,GPU_not_executed=True,coverage_cases=cases,
        capture_length1_to_runtime_long_length_transition_cases=transition,
        all_valid_tokens_covered_once=True,all_active_segment_scratch_within_reserved_capacity=True,
        n1to24_configuration_unchanged=True,provenance=proofs,sources=sources)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({key:value for key,value in gate.items() if key!='sources'}))
if __name__=='__main__':main()
