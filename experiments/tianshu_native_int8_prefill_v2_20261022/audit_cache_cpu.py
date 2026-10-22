# SPDX-License-Identifier: Apache-2.0
"""Validate original qualified cubins and inspect ELF section differences without GPU."""
import base64,hashlib,json,struct
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
qualified=ROOT/'tianshu_m_range_mlp42_tiles_v2_results_20261021/GPU_GATE_32g.json';gate=json.loads(qualified.read_text());cache=Path('/tmp/torchinductor_root/triton/0')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sections(path):
 data=Path(path).read_bytes();assert data[:6]==b'\x7fELF\x02\x01';offset=struct.unpack_from('<Q',data,40)[0];size,count,names_index=struct.unpack_from('<HHH',data,58);assert size==64
 headers=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)];names=headers[names_index];strings=data[names[4]:names[4]+names[5]];out={}
 for header in headers:
  name=strings[header[0]:].split(b'\0',1)[0].decode();payload=data[header[4]:header[4]+header[5]]
  out[name]=dict(type=header[1],flags=header[2],address=header[3],size=header[5],sha256=hashlib.sha256(payload).hexdigest())
 return out
records=[];files={}
for m in (128,512,4096):
 expected=gate['actual_timed_candidate_kernels'][str(m)];directory=base64.b32encode(bytes.fromhex(expected['cache_hash'])).decode().rstrip('=');folder=cache/directory;binary=folder/'_fused.cubin';metadata=json.loads((folder/'_fused.json').read_text())
 assert sha(binary)==expected['binary_sha256'] and metadata['hash']==expected['cache_hash']
 for path in folder.iterdir():
  if path.is_file():files[str(path)]=sha(path)
 records.append(dict(M=m,cubin=str(binary),binary_sha256=sha(binary),cache_hash=metadata['hash'],use_sme=metadata['use_sme']))
old=Path(records[0]['cubin']);new=Path('/root/.triton/cache')/old.parent.name/'_fused.cubin';old_sections,new_sections=sections(old),sections(new)
old_alloc={name:section for name,section in old_sections.items() if section['flags']&2};new_alloc={name:section for name,section in new_sections.items() if section['flags']&2}
assert old_alloc==new_alloc and set(old_alloc)=={'.text','.rodata','.note'} and sha(old)!=sha(new)
result=dict(complete=True,CPU_only=True,GPU_executed=False,qualified_cache_directory=str(cache),qualified_gate_sha256=sha(qualified),qualified_records=records,qualified_cache_files=files,M128_recompiled_cubin=str(new),M128_recompiled_sha256=sha(new),M128_same_cache_key=True,M128_all_allocated_ELF_sections_identical=True,allocated_ELF_sections=old_alloc,nonallocated_different_sections=[name for name in old_sections if old_sections[name]!=new_sections.get(name)],audit_source_sha256=sha(__file__))
(BASE/'QUALIFIED_CACHE_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({key:value for key,value in result.items() if key not in ('qualified_cache_files','qualified_records')}))
