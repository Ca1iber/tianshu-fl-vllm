# SPDX-License-Identifier: Apache-2.0
"""Symmetric uniform signed int8 keys, step1/4; BF16 values and physical layout unchanged."""
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import sys

SOURCE_SHA = 'dc69f525d8c7686a8977f50e901af39820ce92486894890631c53fc01dc46700'
STEP = .25


def encode_bits(bits):
    value = struct.unpack('!f', struct.pack('!I', bits << 16))[0]
    if math.isnan(value):
        return 0
    return round(max(-127., min(127., value * 4.))) & 255


def encode_bits_integer(bits):
    sign, exponent, fraction = bits >> 15, (bits >> 7) & 255, bits & 127
    if exponent == 255 and fraction:
        return 0
    if exponent >= 132:
        magnitude = 127
    elif exponent <= 123:
        magnitude = 0
    else:
        shift = 132 - exponent
        mantissa = 128 + fraction
        magnitude, remainder = divmod(mantissa, 1 << shift)
        half = 1 << (shift - 1)
        magnitude += remainder > half or (remainder == half and magnitude & 1)
        magnitude = min(127, magnitude)
    return (-magnitude if sign else magnitude) & 255


def decode_scalar(code):
    return (code if code < 128 else code - 256) * STEP


def decoded_bits(code):
    return struct.unpack('!I', struct.pack('!f', decode_scalar(code)))[0] >> 16


def render(raw):
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    original = raw.decode()
    node = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == '_cast_kv_tile')
    branch = node.body[1]
    assert isinstance(branch, ast.If)
    lines = original.splitlines(keepends=True)
    before = ''.join(lines[branch.lineno - 1:branch.end_lineno])
    assert 'data.dtype == tl.uint8' in before and '14848' in before
    after = '''    if KV_QUANT_MODE == 0 and data.dtype == tl.uint8:
        return (data.to(tl.int8, bitcast=True).to(tl.float32) * .25).to(Q.dtype)
'''
    text = ''.join(lines[:branch.lineno - 1]) + after + ''.join(lines[branch.end_lineno:])
    assert text.replace(after, before, 1) == original
    compile(text, '<ki8v16_reader>', 'exec')
    return text, dict(original_source_sha256=SOURCE_SHA, generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        only_uint8_decode_branch_changed=True, attention_accumulation_and_integer_offsets_unchanged=True,
        KV_numeric_format_changed=True, K_step=STEP, K_signed_code_min=-127, K_signed_code_max=127,
        nonfinite_policy='NaN=0; infinity saturated; finite values nearest-even then symmetric saturation',
        V_BF16_unchanged=True, model_quality_not_proven=True)


def load_variant(reader, directory):
    text, proof = render(Path(reader.__file__).read_bytes())
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'generated_linear.py'
    path.write_text(text)
    name = '_ki8v16_linear_reader_' + proof['generated_sha256'][:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, dict(proof, generated_source=str(path))
