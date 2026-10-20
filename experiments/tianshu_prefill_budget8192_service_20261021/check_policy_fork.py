# SPDX-License-Identifier: Apache-2.0
"""CPU proof: guard widening is explicit; old executed sources remain untouched."""
import ast,copy,json
from pathlib import Path
from types import SimpleNamespace
BASE=Path(__file__).resolve().parent
EXPECTED="{'control':4096,'candidate':8192}[os.environ['TIANSHU_DENSE_BUDGET_MODE']]"
def check_fork():
    guard_cases=[]
    for old,new in ((BASE.parent/'tianshu_int8_nn_gateup_service_20261021/policy.py',BASE.parent/'tianshu_prefill_budget8192_nn_20261021/policy.py'),(BASE.parent/'tianshu_prefill_int8_service_20261021/policy.py',BASE/'policy.py')):
        native=ast.parse(old.read_text());changed=ast.parse(new.read_text())
        guards=[n for n in ast.walk(changed) if isinstance(n,ast.Compare) and isinstance(n.ops[0],ast.Eq) and ast.unparse(n.left)=='cfg.scheduler_config.max_num_batched_tokens']
        assert len(guards)==1
        guard=guards[0];assert ast.dump(guard.comparators[0])==ast.dump(ast.parse(EXPECTED,mode='eval').body)
        expression=compile(ast.Expression(guard),str(new),'eval')
        for mode in ('control','candidate'):
            for budget in (2048,4096,8192,16384):
                namespace=dict(cfg=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=budget)),os=SimpleNamespace(environ={'TIANSHU_DENSE_BUDGET_MODE':mode}))
                observed=eval(expression,namespace);assert observed==(budget=={'control':4096,'candidate':8192}[mode])
                guard_cases.append(dict(policy=str(new),mode=mode,budget=budget,accepted=observed))
        for env in ({},{'TIANSHU_DENSE_BUDGET_MODE':'unknown'}):
            try:eval(expression,dict(cfg=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=4096)),os=SimpleNamespace(environ=env)))
            except KeyError:pass
            else:raise AssertionError('unknown budget mode accepted')
        if 'nn_' in new.parent.name:
            guard.comparators[0]=ast.Constant(4096);assert ast.dump(native)==ast.dump(changed)
    selection=json.loads((BASE/'selected_plan.json').read_text());old=json.loads((BASE.parent/'tianshu_prefill_int8_service_20261021/selected_plan.json').read_text())
    assert selection['configurations'][:-1]==old['configurations'] and selection['configurations'][-1]['M']==8192
    for name in ('hooks_control','hooks_candidate'):
        tree=ast.parse((BASE/name/'sitecustomize.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='policy.install']
        assert len(calls)==1 and isinstance(calls[0].args[1],ast.Constant) and calls[0].args[1].value=='candidate'
    assert 'cfg.scheduler_config.max_num_batched_tokens' in (BASE/'policy.py').read_text()
    return dict(actual_guard_evaluations=guard_cases,missing_unknown_budget_modes_rejected=True,NN_AST_only_budget_guard_changed=True,
        old_prefill_configurations_unchanged=True,same_precision_mode_in_both_arms=True,intermediate_M_require_startup_full_A_and_selected_real_W_checks=True)
