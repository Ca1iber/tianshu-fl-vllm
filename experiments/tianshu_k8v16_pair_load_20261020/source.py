# SPDX-License-Identifier: Apache-2.0
"""Bit-preserving scalar uint16 and paired-byte K8 load variants."""
import ast, hashlib, importlib.util, sys
from pathlib import Path
PARENT_SHA='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'
MODES=('current','scalar16','pair_rows','pair_columns')
OLD_K='''            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
                cache_modifier=KV_CACHE_MODIFIER if IS_3D else "",
            )'''
HELPER='''@triton.jit
def _paired_soft8_decode(word, high: tl.constexpr):
    if high:
        packed = word >> 8
    else:
        packed = word & 255
    magnitude = packed & 127
    bits = tl.where(magnitude >= 8, (magnitude << 4) + 14848, 0)
    bits = bits | ((packed & 128) << 8)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


'''

def paired_block(mode):
    assert mode in ('pair_rows','pair_columns')
    if mode=='pair_rows':
        offset='''                    physical_block_idx[:, None] * (stride_k_cache_0 // 2)
                    + kv_head_idx * (stride_k_cache_2 // 2)
                    + tl.arange(0, 64)[None, :]
                    + (seq_offset % BLOCK_SIZE)[:, None] * (stride_k_cache_1 // 2)'''
        mask='tile_mask[:, None]'
        joined='tl.join(low, high).reshape(TILE_SIZE, 128).T'
    else:
        offset='''                    physical_block_idx[None, :] * (stride_k_cache_0 // 2)
                    + kv_head_idx * (stride_k_cache_2 // 2)
                    + tl.arange(0, 64)[:, None]
                    + (seq_offset % BLOCK_SIZE)[None, :] * (stride_k_cache_1 // 2)'''
        mask='tile_mask[None, :]'
        joined='tl.join(low, high).permute(0, 2, 1).reshape(128, TILE_SIZE)'
    fallback='\n'.join('    '+line for line in OLD_K.splitlines())
    return f'''            if IS_3D and KV_QUANT_MODE == 0 and HEAD_SIZE == 128 and HEAD_SIZE_PADDED == 128 and stride_k_cache_3 == 1 and key_cache_ptr.dtype.element_ty == tl.uint8:
                pair_offset = (
{offset}
                )
                pair = tl.load(key_cache_ptr.to(tl.pointer_type(tl.uint16)) + pair_offset,
                    mask={mask}, other=0, cache_modifier=KV_CACHE_MODIFIER)
                low = _paired_soft8_decode(pair, False)
                high = _paired_soft8_decode(pair, True)
                K_load = {joined}
            else:
{fallback}'''

def render(parent, mode):
    parent=Path(parent);original=parent.read_text()
    assert hashlib.sha256(original.encode()).hexdigest()==PARENT_SHA
    assert mode in MODES
    generated=original
    pairs=[]
    if mode=='scalar16':
        before='        packed = data.to(tl.uint32)'
        after='        packed = data.to(tl.uint16)'
        pairs=[(before,after)]
    elif mode.startswith('pair_'):
        anchor='@triton.jit\ndef kernel_unified_attention('
        pairs=[(anchor,HELPER+anchor),(OLD_K,paired_block(mode))]
    for before,after in pairs:
        assert generated.count(before)==1
        generated=generated.replace(before,after)
    inverse=generated
    for before,after in reversed(pairs):
        assert inverse.count(after)==1
        inverse=inverse.replace(after,before)
    assert inverse==original
    functions=lambda text:{node.name:ast.dump(node) for node in ast.parse(text).body if isinstance(node,ast.FunctionDef)}
    a,b=functions(original),functions(generated)
    assert set(b)-set(a)==({'_paired_soft8_decode'} if mode.startswith('pair_') else set())
    changed={name for name in a if a[name]!=b[name]}
    assert changed==({'_cast_kv_tile'} if mode=='scalar16' else {'kernel_unified_attention'} if mode.startswith('pair_') else set())
    for name in a:
        if name in changed:
            first=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name==name)
            second=next(n for n in ast.parse(generated).body if isinstance(n,ast.FunctionDef) and n.name==name)
            assert ast.dump(first.args)==ast.dump(second.args)
    return generated,dict(mode=mode,parent_sha256=PARENT_SHA,generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        exact_source_reversal=True,changed_parent_functions=sorted(changed),
        cache_storage_and_V_load_unchanged=True,reducer_AST_unchanged=True,
        all_pointer_offsets_remain_int64=True,only_even_K_strides_and_2byte_alignment_supported=True)

def load(parent,directory,mode):
    text,proof=render(parent,mode);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    path=directory/f'generated_{mode}.py';path.write_text(text)
    name='_pair_load_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module,dict(proof,generated_source=str(path))
