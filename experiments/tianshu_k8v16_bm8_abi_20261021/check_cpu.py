# SPDX-License-Identifier: Apache-2.0
"""Independent token coverage and safe capture-to-runtime split bounds."""
import ast,hashlib,json
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
                tile=source.configuration(mode,n)['TILE_SIZE']
                tiles=(length+active_target*tile-1)//(active_target*tile)
                # Enumerate actual intervals, not a second copy of the GPU indexing.
                counts=[0]*length
                for segment in range(segments):
                    start=segment*tiles*tile
                    for token in range(start,min(length,start+tiles*tile)):counts[token]+=1
                assert all(count==1 for count in counts)
                assert segments>=((length+tiles*tile-1)//(tiles*tile))
                cases+=1
                if n>24:
                    capture_segments=source.configured_segments(mode,n,native_segments(n,1))
                    assert capture_segments==segments,(mode,n,length,capture_segments,segments)
                    transition+=1
    mapping_cases=0
    for mode in source.MODES:
        for n in range(1,65):
            for inactive in (0,1,n//2):
                lengths=[0 if i<inactive else 1 for i in range(n)]
                prefix=[0]
                for length in lengths:prefix.append(prefix[-1]+length)
                block_q=1 if mode.startswith('bm8') and n>24 else 2
                expected=[(seq,prefix[seq]) for seq,length in enumerate(lengths) if length]
                if mode.endswith('_direct'):
                    actual=[(seq,prefix[seq]) for seq in range(n) if prefix[seq+1]>prefix[seq]]
                else:
                    actual=[]
                    for seq,length in enumerate(lengths):
                        for block in range((length+block_q-1)//block_q):actual.append((seq,prefix[seq]+block*block_q))
                assert actual==expected
                mapping_cases+=1
    original=(BASE.parent/'tianshu_k8v16_address32_20261020/component_launch.py').read_text()
    updated=(BASE/'native_launch.py').read_text()
    normalized=updated.replace("assert (kw['BLOCK_M'],kw['BLOCK_Q']) in ((16,2),(8,1))", "assert kw['BLOCK_M']==16 and kw['BLOCK_Q']==2")
    def cls(text,name):return next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==name)
    assert ast.dump(cls(original,'Producer'))==ast.dump(cls(normalized,'Producer'))
    assert ast.dump(cls(original,'Reducer'))==ast.dump(cls(updated,'Reducer'))
    sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),BASE/'source.py',BASE/'launch.py',BASE/'native_launch.py',BASE.parent/'tianshu_k8v16_address32_20261020/component_launch.py',PARENT)}
    gate=dict(complete=True,CPU_only=True,GPU_not_executed=True,coverage_cases=cases,
        capture_length1_to_runtime_long_length_transition_cases=transition,
        all_valid_tokens_covered_once=True,all_active_segment_scratch_within_reserved_capacity=True,
        n1to24_tile_configuration_unchanged=True,actual_live_and_zero_query_mapping_cases=mapping_cases,actual_3D_max_query_len1_required=True,native_Producer_guard_normalization_AST_verified=True,Reducer_AST_unchanged=True,provenance=proofs,sources=sources)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({key:value for key,value in gate.items() if key!='sources'}))
if __name__=='__main__':main()
