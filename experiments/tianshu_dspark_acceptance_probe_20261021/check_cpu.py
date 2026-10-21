# SPDX-License-Identifier: Apache-2.0
"""Prepare matched natural/random diagnostic inputs and pin primary sources."""
import ast,hashlib,json,random
from pathlib import Path
BASE=Path(__file__).resolve().parent;SITE=Path('/usr/local/lib/python3.12/site-packages');REPO=BASE.parents[1]
CPU=Path('/workspace/benchmark-results/20261021_dspark_acceptance_probe_CPU_tianshu');CPU.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p in BASE.rglob('*.py'):ast.parse(p.read_text())
proof=BASE.parent/'tianshu_dspark_input_contract_results_20261021/GPU_GATE.json'
gate=json.loads(proof.read_text());assert gate['complete'] and gate['CPU_reverified'] and gate['cases']==160
for p,h in gate['frozen_sources'].items():assert sha(p)==h
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('/workspace/MiniCPM5-2B',local_files_only=True,trust_remote_code=False)
dataset=Path('/workspace/evalscope-datasets/math_500/test.jsonl')
records=[json.loads(x) for x in dataset.read_text().splitlines() if x.strip()]
natural=[]
for row in records:
 if str(row.get('level','')) not in ('3','Level 3'):continue
 ids=tokenizer.apply_chat_template([dict(role='user',content=row['problem'])],tokenize=True,add_generation_prompt=True,return_dict=False)
 if not 16<=len(ids)<1024:continue
 natural.append(dict(prompt=ids,source_id=str(row.get('unique_id',row.get('id',len(natural)))),input_tokens=len(ids)))
 if len(natural)==8:break
assert len(natural)==8
rng=random.Random(202610210955)
matched=[dict(prompt=[rng.randrange(1,120000) for _ in x['prompt']],source_id=x['source_id'],input_tokens=len(x['prompt'])) for x in natural]
long=[dict(prompt=[rng.randrange(1,120000) for _ in range(4096)],source_id=str(i),input_tokens=4096) for i in range(8)]
workload=dict(natural_math=natural,random_matched=matched,random_4k=long)
(CPU/'inputs.json').write_text(json.dumps(workload,separators=(',',':'))+'\n')
assert [x['input_tokens'] for x in natural]==[x['input_tokens'] for x in matched]
parents=[BASE.parent/'tianshu_dspark_20261012/hooks/sitecustomize.py',BASE.parent/'tianshu_dspark_20261012/tianshu_dspark_adapter/__init__.py',BASE.parent/'tianshu_dspark_20261012/tianshu_dspark_adapter/model.py',BASE.parent/'tianshu_dspark_20261012/tianshu_dspark_adapter/proposer.py',BASE.parent/'tianshu_dspark_20261012/fixed_index.py',BASE.parent/'tianshu_dspark_20261012/rejection_compat.py',BASE.parent/'tianshu_native_prefill_20261012/zero_initialization.py',BASE.parent/'tianshu_dspark_input_contract_20261021/policy.py',proof,BASE.parent/'tianshu_dspark_input_contract_20261021/CPU_GATE.json',Path('/workspace/MiniCPM5-2B/config.json'),Path('/workspace/MiniCPM5-2B-DSpark/config.json'),dataset]
installed=['vllm_fl/__init__.py','vllm_fl/worker/worker.py','vllm_fl/worker/model_runner.py','vllm/v1/attention/backends/triton_attn.py','vllm/config/speculative.py','vllm/model_executor/models/qwen3_dflash.py','vllm/model_executor/models/llama.py','vllm/model_executor/models/interfaces.py','vllm/v1/spec_decode/dflash.py','vllm/v1/spec_decode/llm_base_proposer.py','vllm/v1/spec_decode/utils.py','vllm/v1/core/sched/scheduler.py','vllm/v1/sample/rejection_sampler.py','vllm/v1/sample/ops/topk_topp_sampler.py','vllm/v1/attention/ops/triton_unified_attention.py']
paths=[*BASE.rglob('*.py'),*BASE.glob('*.sh'),BASE/'PLAN.md',*parents,*[SITE/x for x in installed]]
assert all(p.exists() for p in paths),[str(p) for p in paths if not p.exists()]
sources={str(p):sha(p) for p in sorted(set(paths))}
r=dict(complete=True,CPU_only=True,GPU_not_used=True,model_not_loaded=True,official_benchmark=False,accuracy_test=False,input_lengths=[x['input_tokens'] for x in natural],natural_and_random_lengths_matched=True,groups=list(workload),inputs_path=str(CPU/'inputs.json'),inputs_sha256=sha(CPU/'inputs.json'),random_token_range=[1,119999],natural_source='First eight Level3 problems with chat template and 16<=tokens<1024; diagnostics only, no accuracy scoring.',output_tokens=128,temperature=1.0,top_p=.95,ignore_eos=True,concurrency=8,round_orders=[['natural_math','random_matched','random_4k'],['random_4k','random_matched','natural_math']],primary_source_count=len(sources),source_inventory_scope='Explicit primary runtime hooks/proposer/runner/attention/rejection plus CPU evidence and prompt data; not a full transitive Python dependency closure.',sources=sources)
(BASE/'CPU_GATE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('sources',)}))
