# SPDX-License-Identifier: Apache-2.0
import ast,json,os
os.environ['VLLM_PLUGINS']=''
from pathlib import Path
import torch
from observer_profile import BASE,CASES,payload,prepare,digest
from observer import Observer
from vllm.compilation.passes.inductor_pass import pass_context
from vllm.config.utils import Range
root=Path('/workspace/benchmark-results/20261021_nn_silu_fx_observer_CPU_tianshu');assert not root.exists();root.mkdir()
assert not torch.cuda.is_initialized()
for p in BASE.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
for _,salt in CASES:
    body=payload(salt);r=CompletionRequest.model_validate(body);assert r.seed==777 and r.cache_salt==salt and r.return_token_ids
assert len(CASES)==2 and CASES[0][1]!=CASES[1][1]
lib=torch.library.Library('tianshu_prefill_int8','DEF');lib.define('linear(Tensor x, Tensor w, Tensor packed, Tensor scales, int bank_id) -> Tensor')
g=torch.fx.Graph();x=g.placeholder('x');w=g.placeholder('w');packed=g.placeholder('packed');scales=g.placeholder('scales')
z=g.call_function(torch.ops.tianshu_prefill_int8.linear.default,args=(x,w,packed,scales,0));gate=g.call_function(torch.ops.aten.slice.Tensor,args=(z,1,0,6144));up=g.call_function(torch.ops.aten.slice.Tensor,args=(z,1,6144,12288));act=g.call_function(torch.ops.aten.silu.default,args=(gate,));out=g.call_function(torch.ops.aten.mul.Tensor,args=(act,up));g.output(out)
before=g.python_code('self').src;observer=Observer(root/'synthetic_graph')
with pass_context(Range(64,64)):observer(g)
assert g.python_code('self').src==before
record=json.loads(next((root/'synthetic_graph').glob('graph_*.json')).read_text());assert record['no_graph_mutation'] and record['compile_range']==[64,64] and len(record['linear_neighborhoods'])==1
command,environment,profile,sources=prepare(root)
assert len(profile)==62 and len(command)==13 and not torch.cuda.is_initialized()
controllers={str(p):digest(p) for p in [*BASE.rglob('*.py'),BASE/'PLAN_32g.md',BASE/'run_probe.sh']}
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,GPU_not_executed=True,synthetic_graph_read_only_proof=True,synthetic_graph_not_model_graph=True,original_407_sources_exact_subset=True,actual_request_schema_validated=True,source_count=len(sources),sources=sources,controllers=controllers)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','controllers')}))
