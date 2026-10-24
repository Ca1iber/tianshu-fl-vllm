# SPDX-License-Identifier: Apache-2.0
"""Same exact K4 values; load one uint16 word and decode four adjacent D values."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
import k4_bitcast_reader as previous
PARENT_SHA=previous.PARENT_SHA
HELPER='''@triton.jit
def _k4_decode_word(word):
    a = _k4_int_to_bf16((word & 15).to(tl.int16) - 8)
    b = _k4_int_to_bf16(((word >> 4) & 15).to(tl.int16) - 8)
    c = _k4_int_to_bf16(((word >> 8) & 15).to(tl.int16) - 8)
    d = _k4_int_to_bf16((word >> 12).to(tl.int16) - 8)
    return tl.join(tl.join(a, c), tl.join(b, d))


'''
def render(parent):
    base,prior=previous.render(parent)
    start=base.index('            # K4 bytes hold even/odd D nibbles.')
    end=base.index('            # V : (TILE_SIZE, HEAD_SIZE)\n',start)
    old=base[start:end]
    new='''            # One uint16 word holds four adjacent K4 D values.
            word_offset = (
                physical_block_idx[:, None] * (stride_k_cache_0 // 2)
                + kv_head_idx * (stride_k_cache_2 // 2)
                + tl.arange(0, 32)[None, :]
                + (seq_offset % BLOCK_SIZE)[:, None] * (stride_k_cache_1 // 2)
            )
            word = tl.load(key_cache_ptr.to(tl.pointer_type(tl.uint16)) + word_offset,
                mask=tile_mask[:, None], other=34952, cache_modifier=KV_CACHE_MODIFIER)
            K_load = _k4_decode_word(word).reshape(TILE_SIZE, 128).T
'''
    marker='@triton.jit\ndef kernel_unified_attention('
    assert base.count(marker)==base.count(old)==1
    text=base.replace(old,new).replace(marker,HELPER+marker)
    assert text.replace(HELPER+marker,marker).replace(new,old)==base
    definitions=lambda s:{n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    a,b=definitions(base),definitions(text)
    assert set(b)-set(a)=={'_k4_decode_word'}
    assert [n for n in a if ast.dump(a[n])!=ast.dump(b[n])]==['kernel_unified_attention']
    proof=dict(prior);proof.pop('only_added_function',None)
    proof.update(generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        previous_bitcast_generated_sha256=prior['generated_sha256'],exact_inverse_to_bitcast_source=True,
        word_variant_only_added_function='_k4_decode_word',added_functions_relative_to_r5=['_k4_int_to_bf16','_k4_decode_word'],
        word_load_bytes=2,decoded_values_per_word=4,storage_and_numerical_quantization_unchanged=True)
    return text,proof

def load(parent,root):
    text,proof=render(parent);root=Path(root);root.mkdir(exist_ok=False)
    path=root/'generated_k4_reader.py';path.write_text(text)
    name='_K4_word_reader_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
    return m,dict(proof,generated_source=str(path))
