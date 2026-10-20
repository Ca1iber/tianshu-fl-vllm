# SPDX-License-Identifier: Apache-2.0
"""Reuse qualified paired-K reader and native scaled-P path for INT8 V."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
SOURCE=Path('/workspace/benchmark-results/20261020_lm_precision_math_tianshu/pair_load_records/generated_reader/generated_pair_rows.py')
SOURCE_SHA='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'

def render(raw):
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    original=raw.decode();text=original
    edits=[('if KV_QUANT_MODE == 0 and data.dtype == tl.uint8:','if (KV_QUANT_MODE == 0 or KV_QUANT_MODE == 2) and data.dtype == tl.uint8:'),
        ('if IS_3D and KV_QUANT_MODE == 0 and HEAD_SIZE == 128','if IS_3D and (KV_QUANT_MODE == 0 or KV_QUANT_MODE == 2) and HEAD_SIZE == 128')]
    for before,after in edits:assert text.count(before)==1;text=text.replace(before,after)
    restored=text
    for before,after in reversed(edits):assert restored.count(after)==1;restored=restored.replace(after,before)
    assert restored==original
    def function(source,name):return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('reduce_segments','unified_attention'):
        assert ast.dump(function(original,name))==ast.dump(function(text,name))
    return text,dict(parent_source_sha256=SOURCE_SHA,generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        only_two_mode_conditions_extended=True,paired_K_load_enabled_for_both_modes=True,
        producer_arithmetic_and_native_P_scaling_unchanged=True,reducer_and_API_AST_unchanged=True,
        K_fixed_software8_V_per_token_head_INT8=True,K_per_token_head_scales_stored_as_exact_one=True)

def load_modules(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    text,proof=render(SOURCE.read_bytes());file=root/'generated_reader.py';file.write_text(text)
    name='_tianshu_k8vi8_heads_reader';spec=importlib.util.spec_from_file_location(name,file)
    reader=importlib.util.module_from_spec(spec);sys.modules[name]=reader;spec.loader.exec_module(reader)
    return reader,dict(proof,generated_source=str(file))
