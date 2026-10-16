# SPDX-License-Identifier: Apache-2.0
"""Minimal reversible changes to the inspected ModelRunnerFL methods."""
import ast
import hashlib
import textwrap

RUNNER_SHA = 'b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'


def render(raw, mode):
    assert hashlib.sha256(raw).hexdigest() == RUNNER_SHA
    assert mode in ('control', 'int32', 'fused')
    tree = ast.parse(raw.decode())
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'ModelRunnerFL')
    methods = {}
    derivation = dict(mode=mode, original_source_sha256=RUNNER_SHA, reversible=True, methods={})
    for name in ('__init__', '_prepare_inputs'):
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        before = textwrap.dedent('\n'.join(raw.decode().splitlines()[node.lineno-1:node.end_lineno])+'\n')
        after = before
        edits = []
        if name == '__init__' and mode != 'control':
            edits.append(('self.req_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int64)',
                          'self.req_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int32)'))
        if name == '_prepare_inputs' and mode == 'fused':
            edits.append(('''    self.positions[:total_num_scheduled_tokens] = (
        self.num_computed_tokens[req_indices_gpu].to(torch.int64)
        + self.query_pos.gpu[:total_num_scheduled_tokens]
    )
''', '''    if (self.speculative_config is None and not self.uses_mrope
            and self.uses_xdrope_dim == 0):
        _tianshu_positions.launch(self, total_num_scheduled_tokens)
    else:
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )
'''))
        for a, b in edits:
            assert after.count(a) == 1, (name, a)
            after = after.replace(a, b)
        restored = after
        for a, b in reversed(edits):
            assert restored.count(b) == 1
            restored = restored.replace(b, a)
        assert restored == before
        compile('from __future__ import annotations\n'+after, '<request_positions>', 'exec')
        methods[name] = after
        derivation['methods'][name] = dict(edits=len(edits),
            original_sha256=hashlib.sha256(before.encode()).hexdigest(),
            generated_sha256=hashlib.sha256(after.encode()).hexdigest())
    return methods, derivation
