# SPDX-License-Identifier: Apache-2.0
"""Validate selected source binaries and merged model cache without GPU imports."""
import base64,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
cache=Path('/workspace/benchmark-results/20261022_prefill_control_binary_all_GPU_tianshu/qualified_model_cache')
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
gate_path=BASE/'GPU_BINARY_GATE_32g.json';gate=json.loads(gate_path.read_text());assert gate['complete'] and gate['device_AST_equal'] and len(gate['cases'])==46 and len(gate['selected'])==23
m64_path=ROOT/'tianshu_native_int8_gateup_20261022/GPU_GATE_32g.json';m64=json.loads(m64_path.read_text());assert m64['complete'];records=[];files={}
for key,record in sorted(gate['selected'].items(),key=lambda p:int(p[0])):
 expected=record['expected_binary_sha256'];assert record['expected_full_binary_matched'] and record['binary_sha256']==expected;records.append(dict(M=int(key),source='range_'+record['version'],directory=record['compiled_cache_directory'],cache_hash=record['cache_hash'],binary_sha256=expected))
meta=m64['cases'][0]['current_compiled'];directory=base64.b32encode(bytes.fromhex(meta['hash'])).decode().rstrip('=');records.append(dict(M=64,source='qualified_M64_current',directory=str(Path('/root/.triton/cache')/directory),cache_hash=meta['hash'],binary_sha256=meta['binary_sha256']))
for record in records:
 original=Path(record['directory']);copied=cache/original.name
 for folder in (original,copied):
  assert sha(folder/'_fused.cubin')==record['binary_sha256'] and json.loads((folder/'_fused.json').read_text())['hash']==record['cache_hash']
  for path in folder.iterdir():
   if path.is_file():files[str(path)]=sha(path)
for path,digest in gate['sources'].items():assert sha(path)==digest
assert len({p.name for p in cache.iterdir() if p.is_dir()})==24
assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
result=dict(complete=True,CPU_only=True,model_not_started=True,qualified_range_points=23,qualified_M64_points=1,total_strict_binary_points=24,qualified_model_cache_directory=str(cache),records=records,files=files,binary_origin_gate_sha256=sha(gate_path),qualified_M64_component_gate_sha256=sha(m64_path),audit_source_sha256=sha(__file__),group_absolute_source_paths_preserved_and_hashed=True)
(BASE/'QUALIFIED_MODEL_CACHE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({key:value for key,value in result.items() if key not in ('records','files')}))
