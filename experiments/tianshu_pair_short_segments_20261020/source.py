# SPDX-License-Identifier: Apache-2.0
"""Paired-K reader with a controlled short-context split count."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
PARENT_SHA='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
MODES=('current','short_s2','short_s8','short_s16')
SHORT_SPLITS={'current':4,'short_s2':2,'short_s8':8,'short_s16':16}

def configured_segments(mode,n,parent_segments):
    assert mode in MODES and n>0
    return max(parent_segments,SHORT_SPLITS[mode]) if n>24 else parent_segments

def render(parent,mode):
    parent=Path(parent);text=parent.read_text()
    assert hashlib.sha256(text.encode()).hexdigest()==PARENT_SHA and mode in MODES
    old='cdiv_fn(seq_len, 4 * TILE_SIZE)'
    assert text.count(old)==2
    replacement=f'cdiv_fn(seq_len, {SHORT_SPLITS[mode]} * TILE_SIZE)'
    output=text.replace(old,replacement)
    assert output.replace(replacement,old)==text
    functions=lambda value:{n.name:n for n in ast.parse(value).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(text),functions(output);assert a.keys()==b.keys()
    changed={name for name in a if ast.dump(a[name])!=ast.dump(b[name])}
    assert changed==({'kernel_unified_attention','reduce_segments'} if mode!='current' else set())
    assert all(ast.dump(a[name].args)==ast.dump(b[name].args) for name in a)
    return output,dict(mode=mode,parent_sha256=PARENT_SHA,generated_sha256=hashlib.sha256(output.encode()).hexdigest(),
        short_active_segments=SHORT_SPLITS[mode],short_threshold=8192,eligible_batches='num_seqs>24',
        exact_source_reversal=True,changed_functions=sorted(changed),function_ABI_unchanged=True,
        K_codec_cache_format_V_writer_and_other_attention_body_unchanged=True,
        launch_must_reserve_at_least_short_split_count_even_at_capture_length1=True,
        floating_segment_reduction_order_changes=mode!='current',Math_or_service_not_claimed=True)

def load(parent,directory,mode):
    text,proof=render(parent,mode);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    path=directory/f'generated_{mode}.py';path.write_text(text)
    name='_pair_short_segments_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module,dict(proof,generated_source=str(path))
