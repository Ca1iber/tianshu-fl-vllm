# SPDX-License-Identifier: Apache-2.0
import ast,copy,json
from pathlib import Path
from types import SimpleNamespace
BASE=Path(__file__).resolve().parent

def check_guard():
    native=ast.parse((BASE.parent/'tianshu_long_segments_service_20261020/policy.py').read_text())
    changed=ast.parse((BASE/'long_policy.py').read_text())
    def find(tree):return [n for n in ast.walk(tree) if isinstance(n,ast.Assert) and isinstance(n.test,ast.Compare) and ast.unparse(n.test.left)=='cfg.scheduler_config.max_num_batched_tokens']
    original_guard,=find(native);new_guard,=find(changed)
    assert ast.dump(original_guard.test.comparators[0])==ast.dump(ast.Constant(4096))
    normalized=copy.deepcopy(changed);find(normalized)[0].test.comparators[0]=ast.Constant(4096)
    assert ast.dump(native)==ast.dump(normalized)
    expression=compile(ast.Expression(new_guard.test),str(BASE/'long_policy.py'),'eval')
    checks=[]
    for mode in ('control','candidate'):
        for budget in (2048,4096,8192,16384):
            namespace=dict(cfg=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=budget)),os=SimpleNamespace(environ={'TIANSHU_DENSE_BUDGET_MODE':mode}))
            observed=eval(expression,namespace);assert observed==(budget==(4096 if mode=='control' else 8192))
            checks.append(dict(mode=mode,budget=budget,accepted=observed))
    for env in ({},{'TIANSHU_DENSE_BUDGET_MODE':'unknown'}):
        try:eval(expression,dict(cfg=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=4096)),os=SimpleNamespace(environ=env)))
        except KeyError:pass
        else:raise AssertionError('unsupported mode accepted')
    tree=ast.parse((BASE/'hooks/sitecustomize.py').read_text());route=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='route_run')
    calls=[];sentinel=object();replacements=[]
    def spy(*args,**kwargs):calls.append((args,kwargs));return sentinel
    target=(BASE.parent/'tianshu_long_segments_service_20261020/hooks/sitecustomize.py').resolve()
    namespace=dict(Path=Path,BASE=BASE,TARGET=target,REPLACEMENTS=replacements,ORIGINAL_RUN=spy)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[route],type_ignores=[])),str(BASE/'hooks/sitecustomize.py'),'exec'),namespace)
    globals_arg={'test':object()};other=Path('unrelated-relative.py')
    assert namespace['route_run'](other,globals_arg,run_name='probe') is sentinel
    assert calls[-1][0][0] is other and calls[-1][0][1] is globals_arg and calls[-1][1]=={'run_name':'probe'} and not replacements
    assert namespace['route_run'](str(target),globals_arg,run_name='probe') is sentinel
    assert calls[-1][0]==(str(BASE/'long_hooks/sitecustomize.py'),globals_arg) and calls[-1][1]=={'run_name':'probe'} and replacements==[str(target)]
    return dict(whole_long_policy_AST_identical_except_expected_budget=True,actual_guard_executed_for_eight_cases=checks,missing_or_unknown_mode_rejected=True,
        actual_route_function_AST_executed=True,unrelated_path_and_arguments_preserved=True,only_exact_long_hook_redirected=True)
if __name__=='__main__':print(json.dumps(check_guard()))
