# SPDX-License-Identifier: Apache-2.0
"""Add a real encoded KV writer to the previously measured software8 reader."""
import hashlib,importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
LEGACY=BASE.parent/'tianshu_kv_soft8_20261016/codec.py'
VARIANT=BASE.parent/'tianshu_kv_soft8_u32_20261016/codec_variant.py'
WRITER_SHA='f18224e51162e37c14a2e2b3045bca55baf0b207b1d203ad6ee7d8ed417139eb'
assert hashlib.sha256(LEGACY.read_bytes()).hexdigest()=='0ccb887961dd0187abed4d77b9dfd2b3fa083e9e7e79c1650d0444e50838c14b'
spec=importlib.util.spec_from_file_location('_soft8_pipeline_prior',VARIANT);prior=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prior;spec.loader.exec_module(prior)
encode_bits=prior.legacy.encode_bits;decode_scalar=prior.decode_scalar
HELPER='''@triton.jit
def _soft8_encode_bf16(data):
    bits = data.to(tl.uint16, bitcast=True).to(tl.uint32)
    magnitude = bits & 32767
    rounded = (magnitude + 7 + ((magnitude >> 4) & 1)) >> 4
    packed = tl.where(rounded >= 936, tl.minimum(rounded - 928, 127), 0)
    return (packed | ((bits & 32768) >> 8)).to(tl.uint8)


'''
TEST='''\n\n@triton.jit
def _pipeline_encode_all(input_ptr, output_ptr, N: tl.constexpr):
    offsets = tl.program_id(0) * 256 + tl.arange(0,256)
    data = tl.load(input_ptr + offsets, offsets < N, other=0)
    tl.store(output_ptr + offsets, _soft8_encode_bf16(data), offsets < N)
'''

def encode_unsigned(bits):
    magnitude=bits&32767;rounded=(magnitude+7+((magnitude>>4)&1))>>4
    packed=min((rounded-928)&0xffffffff,127) if rounded>=936 else 0
    return packed|((bits&32768)>>8)

def render_writer(raw):
    assert hashlib.sha256(raw).hexdigest()==WRITER_SHA
    original=raw.decode();text=original
    for name in ('key','value'):
        before=f'    else:\n        {name}_tile = {name}_load\n'
        after=f'    else:\n        {name}_tile = _soft8_encode_bf16({name}_load)\n'
        assert text.count(before)==1;text=text.replace(before,after)
    text=text.replace('@triton.jit\ndef reshape_and_cache_kernel_flash(',HELPER+'@triton.jit\ndef reshape_and_cache_kernel_flash(',1)+TEST
    restored=text.removesuffix(TEST).replace(HELPER,'')
    for name in ('key','value'):restored=restored.replace(f'{name}_tile = _soft8_encode_bf16({name}_load)',f'{name}_tile = {name}_load')
    assert restored==original
    return text

def load_modules(cv_source,writer_source,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    reader,proof=prior.load_codec(cv_source,root/'reader')
    text=render_writer(Path(writer_source).read_bytes());file=root/'generated_writer.py';file.write_text(text)
    name='_soft8_pipeline_writer';spec=importlib.util.spec_from_file_location(name,file);writer=importlib.util.module_from_spec(spec);sys.modules[name]=writer;spec.loader.exec_module(writer)
    return reader,writer,dict(reader=proof,writer_path=str(file),writer_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),format_fixed_scale=1/16,not_native_FP8=True)
