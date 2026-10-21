# SPDX-License-Identifier: Apache-2.0
import ast,importlib.util,json,os
os.environ['VLLM_PLUGINS']=''
from pathlib import Path
import torch
from candidate_profile import BASE,prepare,digest,payload
from fx_pass import FusionPass
from vllm.compilation.passes.inductor_pass import pass_context
from vllm.config.utils import Range
root=Path('/workspace/benchmark-results/20261021_nn_silu_fx_service_CPU_v2_tianshu');assert not root.exists();root.mkdir()
assert not torch.cuda.is_initialized()
for p in BASE.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
old=(BASE.parent/'tianshu_int8_nn_gateup_service_20261021/policy.py').read_text()
expected=old.replace("policy_source=str(BASE/'policy.py')","policy_source=str(Path(__file__).resolve())")
expected=expected.replace('    pointers={}',"    globals()['RECORD']=record; pointers={}; globals()['POINTERS']=pointers; globals()['LAYERS']={}")
expected=expected.replace("pointers[actual.data_ptr()]=item;record['banks'].append(item)","pointers[actual.data_ptr()]=item;globals()['LAYERS'][item['bank_id']]=layer;record['banks'].append(item)")
expected=expected.replace('for m in (8,64):assert len','for m in (8,):assert len')
expected=expected.replace('        record.update(completed=True,after_original_capture=True',"        sys.modules['_nn_silu_fx_policy'].validate_capture(record)\n        record.update(completed=True,after_original_capture=True")
assert (BASE/'nn_policy.py').read_text()==expected
for path,module_name in [(BASE/'hooks_nn/sitecustomize.py','_nn_silu_fx_NN_policy'),(BASE/'hooks_prefill/sitecustomize.py','_prefill_int8_service_policy')]:
    text=path.read_text();assert "MODE='candidate'" in text and module_name in text
lib=torch.library.Library('tianshu_prefill_int8','DEF');lib.define('linear(Tensor x, Tensor w, Tensor packed, Tensor scales, int bank_id) -> Tensor')
fused_lib=torch.library.Library('tianshu_nn_silu_fx','DEF');fused_lib.define('fused(Tensor x, Tensor weight, Tensor packed, Tensor scales, int bank_id) -> Tensor')
observer_root=Path('/workspace/benchmark-results/20261021_nn_silu_fx_observer_GPU_tianshu/fx_observer')
records=[json.loads(p.read_text()) for p in observer_root.glob('graph_*.json')]
fragments=[entry for r in records if r['compile_range']==[64,64] for entry in r['linear_neighborhoods']]
assert len(fragments)==42 and sorted(e['linear']['args'][4] for e in fragments)==list(range(42))
def build(entry,m=64,extra_user=False):
    g=torch.fx.Graph();nodes={}
    def enc(v):
        if isinstance(v,dict) and set(v)=={'node'}:
            name=v['node']
            if name not in nodes:nodes[name]=g.placeholder(name)
            return nodes[name]
        if isinstance(v,list):return tuple(enc(x) for x in v)
        if v=='torch.float32':return torch.float32
        if v=='torch.bfloat16':return torch.bfloat16
        return v
    for r in [entry['linear'],*entry['descendants'][:7]]:
        namespace,name,overload=r['target'].split('.');target=getattr(getattr(getattr(torch.ops,namespace),name),overload)
        n=g.call_function(target,args=enc(r['args']),kwargs={});nodes[r['name']]=n
        meta=r['meta'];shape=tuple(m if i==0 else int(v) for i,v in enumerate(meta['shape']));stride=tuple(int(v) for v in meta['stride'])
        n.meta['val']=torch.empty_strided(shape,stride,device='meta',dtype={'torch.float32':torch.float32,'torch.bfloat16':torch.bfloat16}[meta['dtype']])
    out=nodes[entry['descendants'][6]['name']]
    if extra_user:g.call_function(torch.ops.aten.neg.default,args=(nodes[entry['linear']['name']],))
    g.output(out);g.lint();return g
checks=0
for entry in fragments:
  for mode,context,m in [('candidate',Range(64,64),64),('control',Range(64,64),64),('candidate',Range(1,4096),64),('candidate',Range(8,8),8)]:
    graph=build(entry,m);before=graph.python_code('self').src
    observer=FusionPass(mode,root/f"test_{checks:03d}",BASE.parent/'tianshu_nn_silu_two_dot_20261021/kernel.py')
    with pass_context(context):observer(graph)
    if mode=='candidate' and context.start==context.end==64:
        assert sum(str(n.target)=='tianshu_nn_silu_fx.fused.default' for n in graph.nodes)==1
        assert not any(str(n.target) in ('tianshu_prefill_int8.linear.default','aten.sigmoid.default') for n in graph.nodes)
        assert graph.python_code('self').src!=before
    else:assert graph.python_code('self').src==before
    checks+=1
bad=build(fragments[0],extra_user=True);before=bad.python_code('self').src
try:
    with pass_context(Range(64,64)):FusionPass('candidate',root/'reject_extra_user',BASE.parent/'tianshu_nn_silu_two_dot_20261021/kernel.py')(bad)
except AssertionError:pass
else:raise AssertionError('Unexpected extra user accepted')
assert bad.python_code('self').src==before
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
for i in range(64):
    body=payload(f'CPU-{i}',100+i,128);req=CompletionRequest.model_validate(body);assert req.return_token_ids and req.max_tokens==128 and len(req.prompt)==128
assert len({tuple(payload(str(i),100+i,128)['prompt']) for i in range(64)})==64
profiles=[]
for mode in ('control','candidate'):
    target=root/mode;target.mkdir();command,environment,profile,sources=prepare(target,mode);profiles.append((command,profile,sources))
assert profiles[0][0]==profiles[1][0] and set(profiles[0][2])==set(profiles[1][2])
normalized=[{k:v.replace(str(root/mode),'{trial}') for k,v in row[1].items() if k!='PYTHONPATH'} for mode,row in zip(('control','candidate'),profiles)];assert normalized[0]==normalized[1]
assert not torch.cuda.is_initialized()
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,GPU_not_executed=True,actual_observed_42_banks_structurally_reconstructed=True,graph_tests=checks,unexpected_extra_user_rejected_without_mutation=True,non_M64_and_control_graphs_unchanged=True,NN_policy_only_documented_source_changes=True,actual_request_schema_and_64_unique_prompts_validated=True,full_model_numerical_correctness_not_proven=True,source_count=len(profiles[0][2]),sources=profiles[0][2],controllers={str(p):digest(p) for p in [*BASE.rglob('*.py'),BASE/'PLAN_32g.md',BASE/'run_smoke.sh']})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','controllers')}))
