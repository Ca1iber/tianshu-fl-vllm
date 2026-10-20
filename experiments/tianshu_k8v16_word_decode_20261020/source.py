# SPDX-License-Identifier: Apache-2.0
"""Decode each byte to BF16 before assembling the K matrix from uint32 words."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT_SHA='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
MODES=('current','word_rows','word_columns','pair_w2')
SIBLING=BASE.parent/'tianshu_k8v16_pair_load_20261020/source.py'
spec=importlib.util.spec_from_file_location('_word_parent_pair_transform',SIBLING)
pair_source=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair_source)
HELPER='''@triton.jit
def _word_soft8_decode(word, shift: tl.constexpr):
    packed = (word >> shift) & 255
    magnitude = packed & 127
    bits = tl.where(magnitude >= 8, (magnitude << 4) + 14848, 0)
    bits = bits | ((packed & 128) << 8)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


'''

def word_block(mode):
    assert mode in ('word_rows','word_columns')
    if mode=='word_rows':
        offset='''                    physical_block_idx[:, None] * (stride_k_cache_0 // 4)
                    + kv_head_idx * (stride_k_cache_2 // 4)
                    + tl.arange(0, 32)[None, :]
                    + (seq_offset % BLOCK_SIZE)[:, None] * (stride_k_cache_1 // 4)'''
        mask='tile_mask[:, None]'
        joined='tl.join(tl.join(k0, k2), tl.join(k1, k3)).reshape(TILE_SIZE, 128).T'
    else:
        offset='''                    physical_block_idx[None, :] * (stride_k_cache_0 // 4)
                    + kv_head_idx * (stride_k_cache_2 // 4)
                    + tl.arange(0, 32)[:, None]
                    + (seq_offset % BLOCK_SIZE)[None, :] * (stride_k_cache_1 // 4)'''
        mask='tile_mask[None, :]'
        joined='tl.join(tl.join(k0, k2), tl.join(k1, k3)).permute(0, 2, 3, 1).reshape(128, TILE_SIZE)'
    fallback='\n'.join('    '+line for line in pair_source.OLD_K.splitlines())
    return f'''            if IS_3D and KV_QUANT_MODE == 0 and HEAD_SIZE == 128 and HEAD_SIZE_PADDED == 128 and stride_k_cache_3 == 1 and key_cache_ptr.dtype.element_ty == tl.uint8:
                word_offset = (
{offset}
                )
                word = tl.load(key_cache_ptr.to(tl.pointer_type(tl.uint32)) + word_offset,
                    mask={mask}, other=0, cache_modifier=KV_CACHE_MODIFIER)
                k0 = _word_soft8_decode(word, 0)
                k1 = _word_soft8_decode(word, 8)
                k2 = _word_soft8_decode(word, 16)
                k3 = _word_soft8_decode(word, 24)
                K_load = {joined}
            else:
{fallback}'''

def render(parent,mode):
    parent=Path(parent);original=parent.read_text()
    assert hashlib.sha256(original.encode()).hexdigest()==PARENT_SHA
    assert mode in MODES
    pairs=[]
    if mode.startswith('word_'):
        anchor='@triton.jit\ndef kernel_unified_attention('
        pairs=[(anchor,HELPER+anchor),(pair_source.paired_block('pair_rows'),word_block(mode))]
    output=original
    for before,after in pairs:
        assert output.count(before)==1;output=output.replace(before,after)
    inverse=output
    for before,after in reversed(pairs):assert inverse.count(after)==1;inverse=inverse.replace(after,before)
    assert inverse==original
    functions=lambda text:{n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(original),functions(output)
    changed={name for name in a if ast.dump(a[name])!=ast.dump(b[name])}
    assert changed==({'kernel_unified_attention'} if mode.startswith('word_') else set())
    assert set(b)-set(a)==({'_word_soft8_decode'} if mode.startswith('word_') else set())
    assert all(ast.dump(a[name].args)==ast.dump(b[name].args) for name in a)
    return output,dict(mode=mode,parent_sha256=PARENT_SHA,generated_sha256=hashlib.sha256(output.encode()).hexdigest(),
        exact_source_reversal=True,changed_parent_functions=sorted(changed),
        K_cache_V_values_writer_and_reducer_unchanged=True,all_offsets_remain_int64=True,
        K_decode_before_BF16_join=True,launch_num_warps=2 if mode=='pair_w2' else 1,
        requires_4byte_aligned_K_for_word_modes=True)

def load(parent,directory,mode):
    text,proof=render(parent,mode);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    path=directory/f'generated_{mode}.py';path.write_text(text)
    name='_word_decode_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module,dict(proof,generated_source=str(path))
