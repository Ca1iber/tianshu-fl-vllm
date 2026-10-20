# SPDX-License-Identifier: Apache-2.0
"""Isolated K-software8/V-linear-INT8 source preparation; not engine-enabled."""
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import sys

DENOMINATORS = (32, 16, 8)
WRITER_SHA256 = '27a74d3bdfee978780044eb69025c10a21d92a1b0a8cf567ee9fea1e0b8d12b5'


def validate_denominator(denominator):
    if type(denominator) is not int or denominator not in DENOMINATORS:
        raise ValueError('V scale must be exactly one of 1/32, 1/16, 1/8')
    return denominator


def encode_scalar(value, denominator):
    """Round to nearest, ties to even, then saturate signed INT8; NaN -> zero."""
    validate_denominator(denominator)
    if math.isnan(value):
        return 0
    if math.isinf(value):
        return 128 if value < 0 else 127
    return max(-128, min(127, round(value * denominator))) & 255


def decode_scalar(code, denominator):
    validate_denominator(denominator)
    if type(code) is not int or not 0 <= code <= 255:
        raise ValueError('byte code outside [0,255]')
    return (code if code < 128 else code - 256) / denominator


def bf16_scalar(bits):
    return struct.unpack('>f', struct.pack('>I', bits << 16))[0]


def _prior_codec():
    path = Path(__file__).resolve().parents[1] / 'tianshu_kv_soft8_u32_20261016/codec_variant.py'
    spec = importlib.util.spec_from_file_location('_tianshu_linear_v_prior_codec', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render_reader(raw, denominator):
    """Keep K and all attention arithmetic; replace only V's byte cast."""
    denominator = validate_denominator(denominator)
    original, prior_proof = _prior_codec().render_source(raw)
    before = 'V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)'
    after = 'V = _linear_v_cast(V_load, Q, v_scale, KV_QUANT_MODE)'
    anchor = '@triton.jit\ndef _cast_kv_tile('
    assert original.count(before) == original.count(anchor) == 1
    helper = f'''@triton.jit
def _linear_v_cast(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    if KV_QUANT_MODE == 0 and data.dtype == tl.uint8:
        signed = data.to(tl.int8, bitcast=True).to(tl.float32)
        return (signed * {1.0 / denominator!r}).to(Q.dtype)
    return _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE)


'''
    test = '''\n\n@triton.jit
def _linear_v_all_codes(codes_ptr, output_ptr):
    offsets = tl.arange(0, 256)
    data = tl.load(codes_ptr + offsets)
    dummy_q = tl.full([256], 0, tl.bfloat16)
    tl.store(output_ptr + offsets, _linear_v_cast(data, dummy_q, None, 0))
'''
    generated = original.replace(before, after).replace(anchor, helper + anchor) + test
    assert generated.removesuffix(test).replace(helper, '').replace(after, before) == original
    def fn(text, name):
        return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in ('_cast_kv_tile', 'reduce_segments', 'unified_attention'):
        assert ast.dump(fn(original, name)) == ast.dump(fn(generated, name))
    assert ast.dump(fn(original, 'kernel_unified_attention')) == ast.dump(
        fn(generated.replace(after, before), 'kernel_unified_attention'))
    return generated, dict(parent=prior_proof, K_cast_AST_unchanged=True,
        attention_AST_unchanged_except_V_cast=True, reducer_AST_unchanged=True,
        V_storage='uint8 two-complement signed INT8', V_scale=1.0 / denominator,
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest())


def render_writer(raw, denominator):
    denominator = validate_denominator(denominator)
    assert hashlib.sha256(raw).hexdigest() == WRITER_SHA256
    original = raw.decode()
    before = 'value_out = _soft8_encode_bf16(value)'
    after = 'value_out = _linear_v_encode(value)'
    anchor = '@triton.jit\ndef _write('
    assert original.count(before) == original.count(anchor) == 1
    helper = f'''@triton.jit
def _linear_v_encode(value):
    scaled = value.to(tl.float32) * {denominator!r}
    # Clamp before integer conversion; explicitly define NaN/Inf handling.
    scaled = tl.where(scaled == scaled, scaled, 0.0)
    scaled = tl.minimum(tl.maximum(scaled, -128.0), 127.0)
    lower = tl.floor(scaled)
    fraction = scaled - lower
    odd = (lower.to(tl.int32) & 1) != 0
    increment = (fraction > 0.5) | ((fraction == 0.5) & odd)
    rounded = lower.to(tl.int32) + increment.to(tl.int32)
    return rounded.to(tl.int8).to(tl.uint8, bitcast=True)


'''
    test = '''\n\n@triton.jit
def _linear_v_encode_all(X, Y, N: tl.constexpr):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    value = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, _linear_v_encode(value), offsets < N)
'''
    generated = original.replace(before, after).replace(anchor, helper + anchor) + test
    assert generated.removesuffix(test).replace(helper, '').replace(after, before) == original
    return generated, dict(original_sha256=WRITER_SHA256,
        only_V_byte_encoder_replaced=True, K_encoder_unchanged=True,
        slot_addressing_and_negative_slot_handling_unchanged=True,
        V_scale=1.0 / denominator, rounding='nearest ties to even',
        NaN_code=0, negative_infinity_code=128, positive_infinity_code=127,
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest())


def load_modules(reader_path, writer_path, output, denominator):
    """Generate unique modules on the GPU host; does not install any hooks."""
    reader_text, reader_proof = render_reader(Path(reader_path).read_bytes(), denominator)
    writer_text, writer_proof = render_writer(Path(writer_path).read_bytes(), denominator)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    suffix = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    modules = []
    for label, text in (('reader', reader_text), ('writer', writer_text)):
        path = output / f'linear_v_{label}.py'
        path.write_text(text)
        name = f'_tianshu_linear_v_{label}_{denominator}_{suffix}'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return *modules, dict(reader=reader_proof, writer=writer_proof,
        engine_hooks_installed=False, GPU_numerical_status='not established by source generation',
        adoption_allowed=False)