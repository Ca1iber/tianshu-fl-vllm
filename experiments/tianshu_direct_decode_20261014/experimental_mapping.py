# SPDX-License-Identifier: Apache-2.0
"""Reuse native arithmetic; directly map one-query Decode blocks to sequences."""
import hashlib
import importlib.util
from pathlib import Path
import sys

SOURCE_SHA = 'bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0'


def load_kernel(native, output_dir):
    source = Path(native.__file__)
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    text = raw.decode()
    init = "    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)"
    assert text.count(init)==1
    text=text.replace(init,"    if USE_SINKS:\n        L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)\n    else:\n        L = tl.zeros([BLOCK_M], dtype=tl.float32)")
    signature = '    Q_IS_FP8: tl.constexpr = False,\n):'
    assert text.count(signature) == 1
    text = text.replace(signature, '    Q_IS_FP8: tl.constexpr = False,\n    DIRECT_DECODE: tl.constexpr = False,\n):')
    block = '''    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )'''
    assert text.count(block) == 1
    replacement = '''    if DIRECT_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx = 0
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_query_len = tl.load(query_start_len_ptr + seq_idx + 1) - cur_batch_in_all_start_index
        seq_len = tl.load(seq_lens_ptr + seq_idx)
    else:
''' + '\n'.join('    ' + line for line in block.splitlines())
    text = text.replace(block, replacement)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'original_native.py').write_bytes(raw)
    generated = out / 'generated_direct_decode.py'
    generated.write_text(text)
    name = 'tianshu_native_direct_decode'
    spec = importlib.util.spec_from_file_location(name, generated)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.kernel_unified_attention, {
        'original_native_sha256': SOURCE_SHA,
        'generated_sha256': hashlib.sha256(generated.read_bytes()).hexdigest(),
        'generated_source': str(generated),
        'changes': ['new constexpr DIRECT_DECODE', 'direct one-query sequence mapping'],
        'attention_arithmetic_changed_relative_to_L0_baseline': False,
        'baseline_uses_no_sink_L0': True,
        'installed_sources_modified': False}


class KernelRouter:
    def __init__(self, original, candidate):
        self.original = original
        self.candidate = candidate
        self.routes = []

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            q = kwargs.get('query_ptr')
            sequences = kwargs.get('num_seqs', 0)
            supported = (
                not args and kwargs.get('IS_3D') is True and isinstance(grid, tuple)
                and len(grid) == 3 and 1 <= sequences <= 64
                and kwargs.get('BLOCK_Q') == 2 and kwargs.get('num_query_heads') == 16
                and kwargs.get('num_queries_per_kv') == 8 and kwargs.get('HEAD_SIZE') == 128
                and kwargs.get('BLOCK_SIZE') == 16 and q is not None
                and str(q.dtype) == 'torch.bfloat16')
            if not supported:
                return self.original[grid](*args, **kwargs)
            new_grid = (sequences, grid[1], grid[2])
            if (grid, new_grid) not in self.routes:
                self.routes.append((grid, new_grid))
            return self.candidate[new_grid](**kwargs, DIRECT_DECODE=True)
        return invoke
