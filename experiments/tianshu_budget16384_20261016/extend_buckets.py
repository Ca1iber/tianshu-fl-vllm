# SPDX-License-Identifier: Apache-2.0
"""Extend only the maximum budget and bucket list; preserve model guards."""
import ast
import inspect
import textwrap


def extend(module):
    original_points=module.PREFILL_BUCKETS
    assert len(original_points)==75 and max(original_points)==2048
    source=textwrap.dedent(inspect.getsource(module.enabled))
    old='and 1 <= scheduler.max_num_batched_tokens <= 2048'
    assert source.count(old)==1
    replacement=source.replace(old,'and 1 <= scheduler.max_num_batched_tokens <= 16384')
    # Guard against broad constant replacement (hidden_size must remain2048).
    before,after=ast.parse(source),ast.parse(replacement)
    b=[n for n in ast.walk(before) if isinstance(n,ast.Constant)]
    a=[n for n in ast.walk(after) if isinstance(n,ast.Constant)]
    changes=[(x.value,y.value) for x,y in zip(b,a) if x.value!=y.value]
    assert len(b)==len(a) and changes==[(2048,16384)]
    module.PREFILL_BUCKETS=tuple(original_points)+tuple(range(2304,16385,256))
    assert len(module.PREFILL_BUCKETS)==131
    namespace=dict(module.__dict__)
    exec(compile(replacement,'<tianshu_budget16384_enabled>','exec'),namespace)
    # enabled only reads os and config fields; use the existing module globals.
    from types import FunctionType
    function=namespace['enabled']
    module.enabled=FunctionType(function.__code__,module.__dict__,function.__name__)

    def buckets(max_tokens):
        if max_tokens<=0 or max_tokens>16384:
            raise ValueError('Extended budget must fit1..16384')
        return tuple(sorted(set(x for x in module.PREFILL_BUCKETS if x<=max_tokens)|{max_tokens}))

    module.buckets_for_budget=buckets
    return dict(original_points=list(original_points),extended_points=list(module.PREFILL_BUCKETS),
                generated_enabled_source=replacement,changed_constants=changes,
                hidden_size_guard_preserved=True,padded_tokens_function_unmodified=True,
                configure_and_warmup_functions_unmodified=True)
