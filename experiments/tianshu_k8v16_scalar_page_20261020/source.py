# SPDX-License-Identifier: Apache-2.0
"""Load one block-table index when the complete tile is one physical page."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
QUALIFIED_SHA='dc69f525d8c7686a8977f50e901af39820ce92486894890631c53fc01dc46700'
BEFORE='''        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)'''
AFTER='''        if IS_3D and TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.broadcast_to(tl.load(
                block_tables_ptr + block_table_offset + j
            ).to(tl.int64), (TILE_SIZE,))
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)'''

def generate(original):
    assert hashlib.sha256(original.encode()).hexdigest()==QUALIFIED_SHA
    assert original.count(BEFORE)==1
    candidate=original.replace(BEFORE,AFTER)
    assert candidate.count(AFTER)==1 and candidate.replace(AFTER,BEFORE)==original
    def functions(text):
        return {node.name:ast.dump(node) for node in ast.parse(text).body if isinstance(node,ast.FunctionDef)}
    a,b=functions(original),functions(candidate)
    assert a.keys()==b.keys()
    assert {key for key in a if a[key]!=b[key]}=={'kernel_unified_attention'}
    return candidate

def load(reader_path,output):
    reader_path=Path(reader_path)
    generated=generate(reader_path.read_text())
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    path=output/'generated_scalar_page.py'
    path.write_text(generated)
    name='_tianshu_scalar_page_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module,dict(parent_source=str(reader_path),parent_sha256=QUALIFIED_SHA,generated_source=str(path),
                       generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
                       exact_reversal_verified=True,only_block_table_load_changed=True,
                       physical_indices_and_offsets_remain_int64=True)
