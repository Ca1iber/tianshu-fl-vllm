# SPDX-License-Identifier: Apache-2.0
"""Software finite E4M3 codec, exponent bias7 and fixed storage scale1/16."""
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import sys
SOURCE_SHA='8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'
NEEDLE='    if KV_QUANT_MODE == 1:\n        if Q.dtype.is_fp8():'
INSERT='''    if KV_QUANT_MODE == 0 and data.dtype == tl.uint8:
        packed = data.to(tl.uint16)
        magnitude = packed & 127
        bits = tl.where(magnitude >= 8, (magnitude << 4) + 14848, 0)
        bits = bits | ((packed & 128) << 8)
        return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(Q.dtype)
'''
TEST='''\n\n@triton.jit
def _codec_all_codes(codes_ptr, output_ptr):
    offsets = tl.arange(0, 256)
    packed = tl.load(codes_ptr + offsets)
    dummy_q = tl.full([256], 0, tl.bfloat16)
    decoded = _cast_kv_tile(packed, dummy_q, None, 0)
    tl.store(output_ptr + offsets, decoded)
'''


def encode_bits(bits):
    magnitude=bits&32767
    rounded=(magnitude+7+((magnitude>>4)&1))>>4
    code=rounded-928
    code=0 if code<8 else min(127,code)
    return ((bits&32768)>>8)|code


def decode_scalar(code):
    magnitude=code&127
    value=0. if magnitude<8 else (1.+(magnitude%8)/8.)*2.**(magnitude//8-11)
    return math.copysign(value,-1. if code&128 else 1.)


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    original=raw.decode();assert original.count(NEEDLE)==1
    text=original.replace(NEEDLE,INSERT+NEEDLE)+TEST
    assert text.removesuffix(TEST).replace(INSERT+NEEDLE,NEEDLE)==original
    def fn(source,name):return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('kernel_unified_attention','reduce_segments'):
        assert ast.dump(fn(original,name))==ast.dump(fn(text,name))
    return text,dict(original_sha256=SOURCE_SHA,generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        producer_and_reducer_AST_unchanged=True,only_cast_helper_uint8_branch_and_test_added=True,
        fixed_storage_scale=1/16,exponent_bits=4,mantissa_bits=3,exponent_bias=7,subnormal_codes_flush_to_signed_zero=True,
        all_nonzero_exponents_finite=True,largest_magnitude=30.,not_native_FP8_hardware=True)


def load_codec(path,output):
    text,record=render_source(Path(path).read_bytes());output=Path(output);output.mkdir(parents=True,exist_ok=False)
    p=output/'generated_soft8.py';p.write_text(text);name='tianshu_software_fp8'
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module,dict(record,generated_source=str(p))


def check_cpu():
    checked=0
    for bits in range(65536):
        exponent=(bits>>7)&255
        if exponent==255:continue
        value=struct.unpack('>f',struct.pack('>I',bits<<16))[0]
        magnitude=abs(value)
        if magnitude==0.:expected=0
        else:
            fraction,exp=math.frexp(magnitude);e=exp-1
            mantissa=round(magnitude/2.**(e-3))
            if mantissa==16:e+=1;mantissa=8
            exp4=e+11
            expected=0 if exp4<=0 else min(127,exp4*8+mantissa-8)
        expected|=(bits&32768)>>8
        assert encode_bits(bits)==expected,(bits,value,encode_bits(bits),expected)
        checked+=1
    for code in range(256):
        magnitude=code&127
        bits=((magnitude<<4)+14848 if magnitude>=8 else 0)|((code&128)<<8)
        actual=struct.unpack('>f',struct.pack('>I',bits<<16))[0]
        expected=decode_scalar(code)
        assert actual==expected and math.copysign(1.,actual)==math.copysign(1.,expected)
    return dict(complete=True,all65280_finite_BF16_encodings_checked=checked==65280,
        independent_frexp_round_reference=True,all256_decoded_values_and_signed_zeros_checked=True,
        fixed_scale=1/16,subnormals_flushed=True,KV_quantization_changes_values=True,model_GPU_and_Math_not_run=True)


if __name__=='__main__':
    import json
    print(json.dumps(check_cpu()))
