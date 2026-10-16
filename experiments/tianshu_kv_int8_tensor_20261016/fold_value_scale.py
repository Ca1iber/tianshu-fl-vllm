# SPDX-License-Identifier: Apache-2.0
"""Apply per-layer V scale once in the split-KV reducer, not every KV tile."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

CV_SHA = '8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest() == CV_SHA
    text = raw.decode()
    edits = []

    def replace(before, after):
        nonlocal text
        assert text.count(before) == 1, before
        text = text.replace(before, after)
        edits.append((before, after))

    replace('''    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)''', '''    FP8_MAX: tl.constexpr = float8_info.max,
    v_tensor_scale_ptr=None,
    APPLY_V_TENSOR_SCALE: tl.constexpr = False,
):
    query_token_idx = tl.program_id(0)''')
    anchor = '    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)\n'
    replace(anchor, anchor + '''
    if APPLY_V_TENSOR_SCALE:
        tl.static_assert(not USE_FP8)
        acc = acc * tl.load(v_tensor_scale_ptr)
''')
    ast.parse(text)
    restored = text
    for before, after in reversed(edits):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == raw.decode()
    return text, dict(original_sha256=CV_SHA, generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      reversible_to_original=True, replacements=len(edits),
                      producer_unmodified=True, default_reducer_arithmetic_preserved=True)


def load_reducer(source, directory):
    text, record = render_source(Path(source).read_bytes())
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'generated_value_scale.py'
    path.write_text(text)
    name = '_tianshu_kv_value_tensor_scale'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    record['generated_source'] = str(path)
    return module.reduce_segments, record
