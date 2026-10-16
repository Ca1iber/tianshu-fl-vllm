# SPDX-License-Identifier: Apache-2.0
"""Render guarded GPU-only edits to the exact installed ModelRunner method."""
import ast
import hashlib
import textwrap

RUNNER_SHA='b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest()==RUNNER_SHA
    module=ast.parse(raw.decode())
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='ModelRunnerFL')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_inputs')
    original=textwrap.dedent('\n'.join(raw.decode().splitlines()[method.lineno-1:method.end_lineno])+'\n')
    replacements=[]
    def change(before,after):
        nonlocal result
        assert result.count(before)==1,before
        replacements.append((before,after));result=result.replace(before,after)
    result=original
    change('    assert num_reqs > 0\n','    assert num_reqs > 0\n    _tianshu_fast = _tianshu_batch_meta.eligible(self, scheduler_output, num_scheduled_tokens)\n')
    for line in ('self.query_start_loc.copy_to_gpu()',
                 'self.discard_request_mask.copy_to_gpu(num_reqs)',
                 'self.req_indices.copy_to_gpu(total_num_scheduled_tokens)',
                 'self.query_pos.copy_to_gpu(total_num_scheduled_tokens)',
                 'self.num_scheduled_tokens.copy_to_gpu(num_reqs)'):
        change('    '+line+'\n','    if not _tianshu_fast:\n        '+line+'\n')
    change('        self.num_accepted_tokens.gpu.fill_(1)\n',
           '        if not _tianshu_fast:\n            self.num_accepted_tokens.gpu.fill_(1)\n')
    begin=result.index('    self.positions[:total_num_scheduled_tokens] = (')
    end=result.index('\n    self.input_batch.block_table.compute_slot_mapping(',begin)
    before=result[begin:end]
    change(before,'    if _tianshu_fast:\n        _tianshu_batch_meta.launch(self, num_reqs)\n    else:\n'+textwrap.indent(before,'    '))
    before='''    self._prepare_input_ids(
        scheduler_output,
        num_reqs,
        total_num_scheduled_tokens,
        cu_num_tokens,
    )
'''
    change(before,'    if not _tianshu_fast:\n'+textwrap.indent(before,'    '))
    change('        logits_indices = query_start_loc[1:] - 1\n',
           '        logits_indices = self._tianshu_decode_logit_indices[:num_reqs] if _tianshu_fast else query_start_loc[1:] - 1\n')
    restored=result
    for before,after in reversed(replacements):
        assert restored.count(after)==1
        restored=restored.replace(after,before)
    assert restored==original
    compile(result,'generated_decode_prepare.py','exec')
    return result,dict(original_method_sha256=hashlib.sha256(original.encode()).hexdigest(),
        generated_method_sha256=hashlib.sha256(result.encode()).hexdigest(),
        reversible_to_original_method=True,replacements=len(replacements),
        CPU_bookkeeping_unchanged=True,positions_and_req_indices_int64_preserved=True,
        original_KV_block_table_and_slot_mapping_calls_preserved=True)
