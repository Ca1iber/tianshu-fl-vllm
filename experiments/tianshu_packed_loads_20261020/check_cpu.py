# SPDX-License-Identifier: Apache-2.0
"""Independent bytearray/struct address check, including every BF16 bit pattern."""
import argparse,ast,hashlib,importlib.util,json,struct
from pathlib import Path
import source
BASE=Path(__file__).resolve().parent
parent=BASE.parent/'tianshu_long_segments_20261020/source.py'
s=importlib.util.spec_from_file_location('_packed_load_cpu_long_generator',parent)
long_source=importlib.util.module_from_spec(s);s.loader.exec_module(long_source)
reader=Path('/workspace/benchmark-results/20261019_soft8_pipeline_micro_v2_tianshu/micro/codec_generated/reader/generated_soft8_u32.py')
original=long_source.generate(reader.read_text());assert hashlib.sha256(original.encode()).hexdigest()==source.PARENT_SHA
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results');root.mkdir(parents=True,exist_ok=False)
(root/'qualified_S16_reader.py').write_text(original)
generated={}
for mode in ('k32','v32','kv32'):
    output=source.generate(original,mode);ast.parse(output)
    (root/f'generated_{mode}.py').write_text(output)
    generated[mode]=hashlib.sha256(output.encode()).hexdigest()
checks=[]
for factor in (1,2):
    page_bytes=12288;pages=16;guard=page_bytes
    raw=bytearray([17])*((pages*factor+2)*page_bytes)
    for page in range(pages):
        for slot in range(16):
            for head in range(2):
                for dim in range(128):
                    index=((page*16+slot)*2+head)*128+dim
                    base=guard+page*factor*page_bytes
                    raw[base+slot*256+head*128+dim]=index%256
                    struct.pack_into('<H',raw,base+4096+slot*512+head*256+dim*2,index)
    before=hashlib.sha256(raw).hexdigest();keys=values=0
    for logical in range(pages):
        page=pages-1-logical
        for slot in range(16):
            for head in range(2):
                base=guard+page*factor*page_bytes
                for word in range(32):
                    packed=struct.unpack_from('<I',raw,base+slot*256+head*128+word*4)[0]
                    byte=[(packed>>(8*j))&255 for j in range(4)]
                    pairs02,pairs13=(byte[0],byte[2]),(byte[1],byte[3])
                    joined=[(pairs02[i],pairs13[i]) for i in range(2)]
                    for i in range(2):
                        for j in range(2):
                            dim=word*4+i*2+j
                            expected=raw[base+slot*256+head*128+dim]
                            assert joined[i][j]==expected;keys+=1
                for word in range(64):
                    packed=struct.unpack_from('<I',raw,base+4096+slot*512+head*256+word*4)[0]
                    for half in range(2):
                        dim=word*2+half
                        expected=struct.unpack_from('<H',raw,base+4096+slot*512+head*256+dim*2)[0]
                        assert ((packed>>(16*half))&65535)==expected;values+=1
    assert hashlib.sha256(raw).hexdigest()==before and keys==values==65536
    checks.append(dict(page_stride_factor=factor,K_byte_values_verified=keys,V_BF16_bit_patterns_verified=values,
                       block_table_permutation_verified=True,complete_storage_unchanged=True))
for path in BASE.rglob('*.py'):ast.parse(path.read_text())
paths=[*BASE.rglob('*.py'),parent,reader]
result=dict(complete=True,CPU_only=True,GPU_executed=False,adopted=False,format_unchanged=True,
            checks=checks,generated_source_sha256=generated,parent_sha256=source.PARENT_SHA,
            K_cache_stride0_1_2_must_be_multiples_of4=True,V_BF16_stride0_1_2_must_be_even=True,
            pointers_must_be_4byte_aligned=True,only_HEAD_SIZE128_3D_KVmode0_selected=True,
            sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
