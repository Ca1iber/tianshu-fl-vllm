# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json,shutil,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;runner=Path('/workspace/benchmark-results/20261022_native_int8_down42_runner_tianshu');assert not runner.exists();runner.mkdir();cache=runner/'qualified_cache';cache.mkdir()
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
profile=json.loads((ROOT/'tianshu_native_int8_prefill_release_v2_20261022/serve_profile_template.json').read_text());sources=dict(profile['frozen_sources']);sources.update(profile['controller_sources']);assert all(sha(p)==h for p,h in sources.items())
for path in list(BASE.iterdir())+[ROOT/'tianshu_native_int8_lm64_20261022/kernel.py']:
 if path.is_file():
  if path.suffix=='.py':ast.parse(path.read_text())
  sources[str(path)]=sha(path)
qualification=json.loads((ROOT/'tianshu_prefill_control_binary_all_20261022/QUALIFIED_MODEL_CACHE_32g.json').read_text());old=Path(qualification['qualified_model_cache_directory']);assert qualification['complete'] and qualification['total_strict_binary_points']==24;copied={}
for name,digest in qualification['files'].items():
 source=Path(name)
 if source.is_relative_to(old):
  assert sha(source)==digest;target=cache/source.relative_to(old);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert sha(target)==digest;copied[str(target)]=digest
assert len(copied)==192 and json.loads((ROOT/'tianshu_native_int8_down_20261022/INDEPENDENT_CPU_GATE_32g.json').read_text())['complete'];gate=dict(complete=True,CPU_only=True,model_not_started=True,sources=sources,source_count=len(sources),private_cache=str(cache),private_cache_files=copied,new_precision_requires_fresh_quality=True);(BASE/'CPU_PREPARE_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','private_cache_files')}));assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules)
