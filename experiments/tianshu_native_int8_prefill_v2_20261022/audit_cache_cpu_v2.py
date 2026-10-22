# SPDX-License-Identifier: Apache-2.0
import hashlib,json,struct
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
origin=ROOT/'tianshu_prefill_control_binary_20261022/GPU_BINARY_GATE_32g.json';state=json.loads(origin.read_text());assert state['complete'] and state['device_AST_equal'] and set(state['selected'])=={'128','512','4096'}
cache=Path('/workspace/benchmark-results/20261022_native_int8_prefill_v2_runner_tianshu/qualified_cache')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sections(path):
 data=Path(path).read_bytes();assert data[:6]==b'\x7fELF\x02\x01';offset=struct.unpack_from('<Q',data,40)[0];size,count,index=struct.unpack_from('<HHH',data,58);assert size==64;headers=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)];strings=data[headers[index][4]:headers[index][4]+headers[index][5]];result={}
 for h in headers:
  name=strings[h[0]:].split(b'\0',1)[0].decode();result[name]=dict(type=h[1],flags=h[2],address=h[3],size=h[5],sha256=hashlib.sha256(data[h[4]:h[4]+h[5]]).hexdigest())
 return result
files={};records=[]
for m in (128,512,4096):
 record=state['selected'][str(m)];source=Path(record['compiled_cache_directory']);copied=cache/source.name
 for folder in (source,copied):
  assert sha(folder/'_fused.cubin')==record['expected_binary_sha256'];meta=json.loads((folder/'_fused.json').read_text());assert meta['hash']==record['cache_hash']
  for path in folder.iterdir():
   if path.is_file():files[str(path)]=sha(path)
 records.append(record)
assert all(sha(path)==digest for path,digest in state['sources'].items())
old=Path(state['selected']['128']['compiled_cache_directory'])/'_fused.cubin';new=Path('/root/.triton/cache')/old.parent.name/'_fused.cubin';a,b=sections(old),sections(new);allocated_a={k:v for k,v in a.items() if v['flags']&2};allocated_b={k:v for k,v in b.items() if v['flags']&2};assert allocated_a==allocated_b and set(allocated_a)=={'.text','.rodata','.note'} and sha(old)!=sha(new)
gate=dict(complete=True,CPU_only=True,GPU_executed=False,qualified_cache_directory=str(cache),binary_origin_gate_sha256=sha(origin),qualified_records=records,qualified_cache_files=files,binary_origin_sources=state['sources'],M128_same_cache_key=True,M128_all_allocated_ELF_sections_identical=True,allocated_ELF_sections=allocated_a,nonallocated_different_sections=[name for name in a if a[name]!=b.get(name)],audit_source_sha256=sha(__file__))
(BASE/'QUALIFIED_CACHE_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print('Qualified cubins for all three M matched; executable sections audited')
