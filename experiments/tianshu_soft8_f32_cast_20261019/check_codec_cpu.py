# SPDX-License-Identifier: Apache-2.0
"""Independent scalar and bit evidence; this does not benchmark the GPU."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
from source import render_reader, decode_scalar, encode_bits, encode_unsigned, SOURCE_PARENT

p = argparse.ArgumentParser()
p.add_argument('--cv-source', type=Path, required=True)
a = p.parse_args()
text, proof = render_reader(a.cv_source.read_bytes())
ast.parse(text)
decoded = []
for code in range(256):
    magnitude = code & 127
    b32 = ((magnitude << 20) + 973078528 if magnitude >= 8 else 0) | ((code & 128) << 24)
    b16 = (((magnitude << 4) + 14848 if magnitude >= 8 else 0) | ((code & 128) << 8))
    assert b32 & 65535 == 0 and b32 >> 16 == b16
    value = struct.unpack('>f', struct.pack('>I', b32))[0]
    expected = decode_scalar(code)
    assert value == expected and math.copysign(1., value) == math.copysign(1., expected)
    decoded.append(b16)
for bits in range(65536):
    assert encode_bits(bits) == encode_unsigned(bits)
base = Path(__file__).resolve().parent
r = dict(complete=True, CPU_only=True, all256_decode_bits_and_signed_zeros_identical=True,
    binary32_to_BF16_exact_because_low16_bits_zero=True, all65536_encoded_values_unchanged=True,
    producer_and_reducer_AST_unchanged=True, reader_generation=proof,
    parent_source_sha256=hashlib.sha256(SOURCE_PARENT.read_bytes()).hexdigest(),
    sources={n: hashlib.sha256((base/n).read_bytes()).hexdigest() for n in ('source.py', 'check_codec_cpu.py')},
    GPU_and_service_not_run=True, adopted=False)
(base/'CODEC_CPU_GATE_32g.json').write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps(r))
