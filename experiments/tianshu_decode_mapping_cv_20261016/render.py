# SPDX-License-Identifier: Apache-2.0
"""Port the preserved Oct12 direct sequence mapping to the current .cv kernel."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
SOURCE_SHA='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'
BLOCK='''    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )'''
BRANCH='''    if DIRECT_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx = 0
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_query_len = tl.load(query_start_len_ptr + seq_idx + 1) - cur_batch_in_all_start_index
        seq_len = tl.load(seq_lens_ptr + seq_idx)
    else:
'''+ '\n'.join('    '+line for line in BLOCK.splitlines())
SIGNATURE='    KV_CACHE_MODIFIER: tl.constexpr = "",\n):'
NEW_SIGNATURE='    KV_CACHE_MODIFIER: tl.constexpr = "",\n    DIRECT_DECODE: tl.constexpr = False,\n):'


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    text=raw.decode();assert text.count(SIGNATURE)==text.count(BLOCK)==1
    generated=text.replace(SIGNATURE,NEW_SIGNATURE).replace(BLOCK,BRANCH)
    assert generated.replace(NEW_SIGNATURE,SIGNATURE).replace(BRANCH,BLOCK)==text
    def fn(source,name):return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(fn(text,'reduce_segments'))==ast.dump(fn(generated,'reduce_segments'))
    return generated,dict(original_sha256=SOURCE_SHA,generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        only_two_reversible_edits=True,reducer_AST_unchanged=True,attention_loop_arithmetic_unchanged=True,
        old_experiment='experiments/tianshu_decode_mapping_20261012',producer_only=True)


def load_kernel(path,output):
    path=Path(path);text,record=render_source(path.read_bytes());output=Path(output);output.mkdir(parents=True,exist_ok=False)
    p=output/'generated_direct_decode.py';p.write_text(text)
    name='tianshu_cv_direct_decode';spec=importlib.util.spec_from_file_location(name,p)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module.kernel_unified_attention,dict(record,generated_source=str(p))
