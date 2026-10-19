# SPDX-License-Identifier: Apache-2.0
"""Change only one-query CTA mapping; preserve the qualified byte-reader arithmetic."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

SOURCE_SHA='dc69f525d8c7686a8977f50e901af39820ce92486894890631c53fc01dc46700'

def render(raw):
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    text=raw.decode()
    signature='    KV_CACHE_MODIFIER: tl.constexpr = "",\n):'
    assert text.count(signature)==1
    text=text.replace(signature,'    KV_CACHE_MODIFIER: tl.constexpr = "",\n    DIRECT_DECODE: tl.constexpr = False,\n):')
    block='''    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )'''
    assert text.count(block)==1
    replacement='''    if DIRECT_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx = 0
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_query_len = tl.load(query_start_len_ptr + seq_idx + 1) - cur_batch_in_all_start_index
        seq_len = tl.load(seq_lens_ptr + seq_idx)
    else:
'''+ '\n'.join('    '+line for line in block.splitlines())
    text=text.replace(block,replacement)
    original,reverse=ast.parse(raw),ast.parse(text)
    kernel=next(n for n in reverse.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_unified_attention')
    assert kernel.args.args[-1].arg=='DIRECT_DECODE' and kernel.args.defaults[-1].value is False
    kernel.args.args.pop();kernel.args.defaults.pop()
    i=next(i for i,n in enumerate(kernel.body) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='DIRECT_DECODE')
    kernel.body[i:i+1]=kernel.body[i].orelse
    assert ast.dump(original)==ast.dump(reverse),'Unexpected arithmetic or source change'
    return text

def load(reader,directory):
    source=Path(reader.__file__);raw=source.read_bytes();text=render(raw)
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'original_reader.py').write_bytes(raw)
    path=directory/'generated_direct_reader.py';path.write_text(text)
    name='_tianshu_k8v16_direct_mapping_reader_20261019'
    assert name not in sys.modules
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module,dict(original_reader_path=str(source),original_sha256=SOURCE_SHA,
        generated_source=str(path),generated_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        AST_identical_after_reversing_only_signature_and_resolver_branch=True,
        attention_arithmetic_unchanged=True,installed_sources_modified=False)
