# SPDX-License-Identifier: Apache-2.0
"""Same K4 numerical contract; exact small integers -> BF16 bits, no sitofp."""
import ast,hashlib,importlib.util,sys
from pathlib import Path
import numpy as np
import k4_reader as original
PARENT_SHA=original.PARENT_SHA
HELPER='''@triton.jit
def _k4_int_to_bf16(q):
    magnitude = tl.abs(q).to(tl.uint16)
    bits = tl.where(magnitude >= 4, 16384 + (magnitude << 5),
        tl.where(magnitude >= 2, 16256 + (magnitude << 6),
        tl.where(magnitude == 1, 16256, 0))).to(tl.uint16)
    bits = bits | tl.where(q < 0, 32768, 0).to(tl.uint16)
    return bits.to(tl.bfloat16, bitcast=True)


'''

def cpu_bits(q):
    q=np.asarray(q,dtype=np.int16);m=np.abs(q).astype(np.uint16)
    b=np.where(m>=4,16384+(m.astype(np.uint32)<<5),
        np.where(m>=2,16256+(m.astype(np.uint32)<<6),np.where(m==1,16256,0))).astype(np.uint16)
    return b|np.where(q<0,32768,0).astype(np.uint16)

def render(parent):
    base,prior=original.render(parent)
    a='            low = ((packed & 15).to(tl.int16) - 8).to(tl.bfloat16)\n'
    b='            high = ((packed >> 4).to(tl.int16) - 8).to(tl.bfloat16)\n'
    aa='            low = _k4_int_to_bf16((packed & 15).to(tl.int16) - 8)\n'
    bb='            high = _k4_int_to_bf16((packed >> 4).to(tl.int16) - 8)\n'
    marker='@triton.jit\ndef kernel_unified_attention('
    assert all(base.count(x)==1 for x in (a,b,marker))
    text=base.replace(a,aa).replace(b,bb).replace(marker,HELPER+marker)
    inverse=text.replace(HELPER+marker,marker).replace(aa,a).replace(bb,b);assert inverse==base
    defs=lambda s:{n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    before,after=defs(base),defs(text)
    assert set(after)-set(before)=={'_k4_int_to_bf16'}
    assert [n for n in before if ast.dump(before[n])!=ast.dump(after[n])]==['kernel_unified_attention']
    assert ast.dump(before['kernel_unified_attention'].args)==ast.dump(after['kernel_unified_attention'].args)
    return text,dict(prior,generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        first_version_generated_sha256=prior['generated_sha256'],exact_inverse_to_first_K4_source=True,
        only_added_function='_k4_int_to_bf16',integer_to_BF16_via_exact_bits=True,
        unchanged_K4_storage_quantization_and_score_arithmetic=True)

def load(parent,root):
    text,proof=render(parent);root=Path(root);root.mkdir(exist_ok=False)
    path=root/'generated_k4_reader.py';path.write_text(text)
    name='_K4_bitcast_reader_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m)
    return m,dict(proof,generated_source=str(path))
