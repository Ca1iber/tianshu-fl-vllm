# SPDX-License-Identifier: Apache-2.0
"""Extend only the real K8V16 wrapper's small-query 3D eligibility; no GPU imports."""
import ast
import hashlib
from pathlib import Path

PARENT = Path('/workspace/benchmark-results/20261020_k8v16_pair_load_GPU_tianshu/pair_rows_reader/generated_pair_rows.py')
PARENT_SHA = 'f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
OLD = '        or max_seqlen_q > 1\n'
NEW = '        or max_seqlen_q > 2\n'


def generate():
    original = PARENT.read_text()
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
    assert original.count(OLD) == 1 and NEW not in original
    generated = original.replace(OLD, NEW)
    assert generated.replace(NEW, OLD) == original
    before, after = ast.parse(original), ast.parse(generated)
    before_kernel = [ast.dump(x) for x in before.body if isinstance(x, ast.FunctionDef) and x.name != 'unified_attention']
    after_kernel = [ast.dump(x) for x in after.body if isinstance(x, ast.FunctionDef) and x.name != 'unified_attention']
    assert before_kernel == after_kernel
    return generated, dict(parent=str(PARENT), parent_sha256=PARENT_SHA,
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        sole_change='wrapper 3D max_seqlen_q upper bound 1 -> 2',
        all_device_function_AST_unchanged=True, software_K8_V_BF16_codec_unchanged=True)
