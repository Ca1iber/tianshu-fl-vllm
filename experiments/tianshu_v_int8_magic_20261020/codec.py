# SPDX-License-Identifier: Apache-2.0
"""Same two-complement V bytes; decode by exact FP32 bit construction and FMA."""
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent / 'tianshu_v_int8_20261020/codec.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == '9ecb4a6671166965e79e74f94f265be2edd29b3d4d2f0b0c97e8f8a1caed4459'
spec = importlib.util.spec_from_file_location('_linear_v_fixed_format_parent', PARENT)
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
DENOMINATORS = parent.DENOMINATORS
WRITER_SHA256 = parent.WRITER_SHA256
validate_denominator = parent.validate_denominator
encode_scalar, decode_scalar = parent.encode_scalar, parent.decode_scalar
bf16_scalar, _prior_codec = parent.bf16_scalar, parent._prior_codec
render_writer = parent.render_writer


def magic_scalar(code, denominator):
    validate_denominator(denominator)
    assert type(code) is int and 0 <= code <= 255
    bits = ((code ^ 128) | 0x4b000000)
    reconstructed = struct.unpack('>f', struct.pack('>I', bits))[0]
    return reconstructed * (1.0 / denominator) - 8388736.0 / denominator


def render_reader(raw, denominator):
    original, proof = parent.render_reader(raw, denominator)
    before = f'''        signed = data.to(tl.int8, bitcast=True).to(tl.float32)
        return (signed * {1.0 / denominator!r}).to(Q.dtype)
'''
    after = f'''        biased = data.to(tl.uint32) ^ 128
        bits = biased | 0x4b000000
        reconstructed = bits.to(tl.float32, bitcast=True)
        return tl.fma(reconstructed, {1.0 / denominator!r}, {-8388736.0 / denominator!r}).to(Q.dtype)
'''
    assert original.count(before) == 1
    generated = original.replace(before, after)
    assert generated.replace(after, before) == original
    for code in range(256):
        assert magic_scalar(code, denominator) == decode_scalar(code, denominator)
    return generated, dict(proof, previous_linear_reader_sha256=proof['generated_sha256'],
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        all256_magic_codepoints_equal_signed_INT8=True, changed_only_V_decode_helper=True,
        byte_format_writer_scale_K_and_attention_arithmetic_unchanged=True,
        decode_method='uint32 xor/or, FP32 bitcast and exact affine FMA',
        exactness='2^23 + (code xor 128) is exact FP32; power-of-two scale and subtraction are exact for all 256 codes')


def load_modules(reader_path, writer_path, output, denominator):
    reader_text, reader_proof = render_reader(Path(reader_path).read_bytes(), denominator)
    writer_text, writer_proof = render_writer(Path(writer_path).read_bytes(), denominator)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    suffix = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    modules = []
    for label, text in (('reader', reader_text), ('writer', writer_text)):
        path = output / f'magic_v_{label}.py'
        path.write_text(text)
        name = f'_tianshu_magic_v_{label}_{denominator}_{suffix}'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return *modules, dict(reader=reader_proof, writer=writer_proof,
        engine_hooks_installed=False, GPU_numerical_status='not established by source generation',
        adoption_allowed=False)
