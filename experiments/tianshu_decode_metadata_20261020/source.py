# SPDX-License-Identifier: Apache-2.0
"""Extract one native block and prepare a strictly guarded replacement branch."""
import ast
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
NATIVE = Path('/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py')
NATIVE_SHA = 'b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'


def extract():
    assert hashlib.sha256(NATIVE.read_bytes()).hexdigest() == NATIVE_SHA
    module = ast.parse(NATIVE.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'ModelRunnerFL')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_prepare_inputs')
    start = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == 'self.req_indices.np[:total_num_scheduled_tokens]')
    end = next(i for i, node in enumerate(function.body) if i >= start
        and ast.unparse(node) == 'self.seq_lens[num_reqs:].fill_(0)') + 1
    block = copy.deepcopy(function.body[start:end])
    assert ast.unparse(block[1]) == 'self.req_indices.copy_to_gpu(total_num_scheduled_tokens)'
    assert ast.unparse(block[-1]) == 'self.seq_lens[num_reqs:].fill_(0)'
    return function, start, end, block


def baseline_function():
    _, _, _, block = extract()
    function = ast.parse('def baseline(self, req_indices, num_scheduled_tokens, num_reqs, total_num_scheduled_tokens):\n    pass\n').body[0]
    function.body = block
    space = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(NATIVE), 'exec'), space)
    return space['baseline']


def eligible(owner, scheduled, n, total):
    if (not 1 <= n <= 64 or total != n or scheduled.shape != (n,)
        or not np.all(scheduled == 1) or owner.uses_mrope or owner.uses_xdrope_dim > 0
        or owner.speculative_config is not None or owner.enable_prompt_embeds
        or owner.mamba_prev_last_scheduled_idx is not None or owner.use_async_spec_decode):
        return False
    tensors = [(owner.num_computed_tokens, torch.int32), (owner.req_indices.gpu, torch.int64),
               (owner.query_pos.gpu, torch.int64), (owner.num_scheduled_tokens.gpu, torch.int32),
               (owner.positions, torch.int64), (owner.seq_lens, torch.int32)]
    device = owner.num_computed_tokens.device
    if device.type != 'cuda':
        return False
    for tensor, dtype in tensors:
        if tensor.dtype != dtype or tensor.device != device or tensor.ndim != 1 or tensor.stride(0) != 1:
            return False
        if tensor.numel() < n:
            return False
    if owner.seq_lens.numel() > owner.num_computed_tokens.numel():
        return False
    return True


def render():
    original, start, end, block = extract()
    selected = copy.deepcopy(original)
    branch = ast.parse('''if _tianshu_decode_metadata_selected(self, scheduler_output, num_scheduled_tokens, num_reqs, total_num_scheduled_tokens):
    self.req_indices.np[:total_num_scheduled_tokens] = req_indices
    self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
    _tianshu_decode_metadata_launch(self, num_reqs)
    req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]
    num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
else:
    pass
''').body[0]
    branch.orelse = block
    selected.body[start:end] = [branch]
    # Independently restore the sole replacement and prove all other AST is identical.
    restored = copy.deepcopy(selected)
    restored.body[start:start + 1] = copy.deepcopy(block)
    assert ast.dump(original, include_attributes=False) == ast.dump(restored, include_attributes=False)
    text = 'from __future__ import annotations\n' + ast.unparse(ast.fix_missing_locations(selected)) + '\n'
    return text, dict(native_source=str(NATIVE), native_source_sha256=NATIVE_SHA,
        native_replaced_statements=len(block), only_selected_AST_block_replaced=True,
        all_other_native_statements_and_signature_unchanged=True,
        native_unselected_block_AST_unchanged=True, positions_and_metadata_only=True,
        RNG_KV_and_model_not_accessed=True)
