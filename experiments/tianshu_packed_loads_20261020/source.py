# SPDX-License-Identifier: Apache-2.0
"""Aligned word loads for the existing K8/VBF16 bits; no cache-format or arithmetic change."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
PARENT_SHA='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'
OLD_K='''            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
                cache_modifier=KV_CACHE_MODIFIER if IS_3D else "",
            )'''
NEW_K='''            if IS_3D and KV_QUANT_MODE == 0 and HEAD_SIZE == 128 and HEAD_SIZE_PADDED == 128 and stride_k_cache_3 == 1 and key_cache_ptr.dtype.element_ty == tl.uint8:
                k_word_offset = (
                    physical_block_idx[None, :] * (stride_k_cache_0 // 4)
                    + kv_head_idx * (stride_k_cache_2 // 4)
                    + tl.arange(0, 32)[:, None]
                    + (seq_offset % BLOCK_SIZE)[None, :] * (stride_k_cache_1 // 4)
                )
                k_words = tl.load(key_cache_ptr.to(tl.pointer_type(tl.uint32)) + k_word_offset,
                    mask=tile_mask[None, :], other=0, cache_modifier=KV_CACHE_MODIFIER)
                k_byte0 = (k_words & 255).to(tl.uint8)
                k_byte1 = ((k_words >> 8) & 255).to(tl.uint8)
                k_byte2 = ((k_words >> 16) & 255).to(tl.uint8)
                k_byte3 = (k_words >> 24).to(tl.uint8)
                K_load = tl.join(tl.join(k_byte0, k_byte2), tl.join(k_byte1, k_byte3)).permute(0, 2, 3, 1).reshape(128, TILE_SIZE)
            else:
                K_load = tl.load(
                    key_cache_ptr + k_offset,
                    mask=dim_mask[:, None] & tile_mask[None, :],
                    other=0.0,
                    cache_modifier=KV_CACHE_MODIFIER if IS_3D else "",
                )'''
OLD_V='''            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
                cache_modifier=KV_CACHE_MODIFIER if IS_3D else "",
            )'''
NEW_V='''            if IS_3D and KV_QUANT_MODE == 0 and HEAD_SIZE == 128 and HEAD_SIZE_PADDED == 128 and stride_v_cache_3 == 1 and value_cache_ptr.dtype.element_ty == tl.bfloat16:
                v_word_offset = (
                    physical_block_idx[:, None] * (stride_v_cache_0 // 2)
                    + kv_head_idx * (stride_v_cache_2 // 2)
                    + tl.arange(0, 64)[None, :]
                    + (seq_offset % BLOCK_SIZE)[:, None] * (stride_v_cache_1 // 2)
                )
                v_words = tl.load(value_cache_ptr.to(tl.pointer_type(tl.uint32)) + v_word_offset,
                    mask=tile_mask[:, None], other=0, cache_modifier=KV_CACHE_MODIFIER)
                v_low = (v_words & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True)
                v_high = (v_words >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
                V_load = tl.join(v_low, v_high).reshape(TILE_SIZE, 128)
            else:
                V_load = tl.load(
                    value_cache_ptr + v_offset,
                    mask=dim_mask[None, :] & tile_mask[:, None],
                    other=0.0,
                    cache_modifier=KV_CACHE_MODIFIER if IS_3D else "",
                )'''

def generate(original,mode):
    assert hashlib.sha256(original.encode()).hexdigest()==PARENT_SHA
    assert mode in ('k32','v32','kv32')
    pairs=[]
    if mode in ('k32','kv32'):pairs.append((OLD_K,NEW_K))
    if mode in ('v32','kv32'):pairs.append((OLD_V,NEW_V))
    output=original
    for before,after in pairs:
        assert output.count(before)==1;output=output.replace(before,after)
    reversed_source=output
    for before,after in reversed(pairs):
        assert reversed_source.count(after)==1;reversed_source=reversed_source.replace(after,before)
    assert reversed_source==original
    functions=lambda text:{node.name:ast.dump(node) for node in ast.parse(text).body if isinstance(node,ast.FunctionDef)}
    a,b=functions(original),functions(output);assert a.keys()==b.keys()
    assert {name for name in a if a[name]!=b[name]}=={'kernel_unified_attention'}
    return output

def load(reader_path,directory,mode):
    reader_path=Path(reader_path);text=generate(reader_path.read_text(),mode)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    path=directory/f'generated_packed_{mode}.py';path.write_text(text)
    name='_tianshu_packed_load_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module,dict(parent_source=str(reader_path),parent_sha256=PARENT_SHA,generated_source=str(path),
                       generated_sha256=hashlib.sha256(text.encode()).hexdigest(),mode=mode,exact_reversal_verified=True,
                       only_K_or_V_load_changed=True,existing_S16_logic_and_function_ABI_unchanged=True,
                       requires_aligned_cache_strides_and_pointer=True,offsets_remain_int64=True,
                       storage_bytes_and_KV_values_not_requantized=True)
