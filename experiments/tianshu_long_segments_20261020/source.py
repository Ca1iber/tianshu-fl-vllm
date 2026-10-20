# SPDX-License-Identifier: Apache-2.0
"""Keep four active segments below 8193 tokens; use captured maximum above it.

Only the two tile-count expressions change. Constant denominators on both sides
of tl.where avoid adding a variable division to the producer's segment setup.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
QUALIFIED_SHA='dc69f525d8c7686a8977f50e901af39820ce92486894890631c53fc01dc46700'
REPLACEMENTS=(
    ('        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)',
     '        tiles_per_segment = tl.where((num_seqs > 24) & (seq_len <= 8192), cdiv_fn(seq_len, 4 * TILE_SIZE), cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE))'),
    ('    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)',
     '    tiles_per_segment = tl.where((num_seqs > 24) & (seq_len <= 8192), cdiv_fn(seq_len, 4 * TILE_SIZE), cdiv_fn(seq_len, num_segments * TILE_SIZE))'),
)

def generate(original):
    assert hashlib.sha256(original.encode()).hexdigest()==QUALIFIED_SHA
    generated=original
    for before,after in REPLACEMENTS:
        assert generated.count(before)==1
        generated=generated.replace(before,after)
    reverse=generated
    for before,after in reversed(REPLACEMENTS):
        assert reverse.count(after)==1
        reverse=reverse.replace(after,before)
    assert reverse==original
    before_tree,after_tree=ast.parse(original),ast.parse(generated)
    before_functions={node.name:node for node in before_tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    after_functions={node.name:node for node in after_tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    assert before_functions.keys()==after_functions.keys()
    changed={name for name in before_functions if ast.dump(before_functions[name])!=ast.dump(after_functions[name])}
    assert changed=={'kernel_unified_attention','reduce_segments'},changed
    assert ast.dump(before_functions['_cast_kv_tile'])==ast.dump(after_functions['_cast_kv_tile'])
    return generated

def load(reader_path,output):
    reader_path=Path(reader_path)
    original=reader_path.read_text()
    generated=generate(original)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    path=output/'generated_dynamic_segments.py'
    path.write_text(generated)
    name='_tianshu_dynamic_segments_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module,dict(parent_source=str(reader_path),parent_sha256=QUALIFIED_SHA,generated_source=str(path),
                       generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
                       exact_reversal_verified=True,only_two_tile_count_expressions_changed=True,
                       KV_decode_and_function_ABI_unchanged=True,threshold_tokens=8192,minimum_batch=25)
