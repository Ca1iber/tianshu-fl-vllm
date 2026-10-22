# SPDX-License-Identifier: Apache-2.0
import ast,csv,json,sys
from pathlib import Path
from candidate_profile import BASE,EXP,prepare,digest,read,frozen_check
root=Path('/workspace/benchmark-results/20261022_native_int8_down_service_CPU_tianshu');assert not root.exists();root.mkdir()
holdout=EXP/'tianshu_native_int8_down42_20261022';gate=read(holdout/'INDEPENDENT_CPU_GATE_32g.json');raw=read(Path(gate['raw_directory'])/'result.json');assert gate['complete'] and raw['complete'] and gate['raw_result_sha256']==digest(Path(gate['raw_directory'])/'result.json') and gate['key_timings_sha256']==digest(holdout/'KEY_TIMINGS_32g.csv')
rows=list(csv.DictReader((holdout/'KEY_TIMINGS_32g.csv').open()));assert any(r['scope']=='whole_MLP' and float(r['speedup_pct'])>=10 for r in rows)
banks=[dict(layer=r['layer'],prefix=f"model.layers.{r['layer']}.mlp.down_proj",original_BF16_sha256=r['Down_original_BF16_sha256'],packed_sha256=r['Down_packed_sha256'],scales_sha256=r['Down_scales_sha256']) for r in raw['banks']];selection=dict(complete=True,quality_not_yet_established=True,holdout_authorizes_fresh_quality_only=True,block_by_M={'64':1024,'128':1024,'512':1024,'4096':512},expected_dequant_binaries={str(c['M']):c['candidate_compiled']['binary_sha256'] for c in raw['cases']},banks=banks,holdout_gate_sha256=digest(holdout/'INDEPENDENT_CPU_GATE_32g.json'),holdout_raw_result_sha256=gate['raw_result_sha256']);(BASE/'selected_plan.json').write_text(json.dumps(selection,indent=2)+'\n')
for p in BASE.rglob('*.py'):ast.parse(p.read_text())
results=[];canonical=None
for mode,scope in [('candidate','all'),('candidate','prefill'),('control','all')]:
 out=root/f'{scope}_{mode}';out.mkdir();command,environment,profile,sources=prepare(out,mode,scope,check_gate=False);assert len(command)==13 and len(profile)==62 and len({**profile,'TRITON_CACHE_DIR':environment['TRITON_CACHE_DIR']})==63
 if canonical is None:canonical=sources
 else:assert canonical==sources
 results.append(dict(mode=mode,scope=scope,command=command,profile_environment=profile,TRITON_CACHE_DIR=environment['TRITON_CACHE_DIR']))
frozen_check(canonical);own=dict(complete=True,CPU_only=True,model_not_started=True,sources=canonical,source_count=len(canonical),all_original877_sources_retained=True,profiles=results);(BASE/'CPU_GATE_32g.json').write_text(json.dumps(own,indent=2)+'\n');assert not any(name.split('.')[0] in ('torch','triton','vllm','flag_gems') for name in sys.modules);print(json.dumps({k:v for k,v in own.items() if k not in ('sources','profiles')}))
