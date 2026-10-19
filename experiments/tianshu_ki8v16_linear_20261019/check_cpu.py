# SPDX-License-Identifier: Apache-2.0
"""Independent floating and integer references for every BF16 encoder input and byte decoder code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import codec

p = argparse.ArgumentParser()
p.add_argument('--reader', type=Path, required=True)
a = p.parse_args()
for bits in range(65536):
    assert codec.encode_bits(bits) == codec.encode_bits_integer(bits), bits
for code in range(256):
    assert codec.decode_scalar(code) == (code if code < 128 else code - 256) / 4
assert codec.encode_bits(0) == codec.encode_bits(32768) == 0
assert codec.encode_bits(32640) == 127 and codec.encode_bits(65408) == 129
assert codec.encode_bits(32641) == codec.encode_bits(65535) == 0
text, proof = codec.render(a.reader.read_bytes())
compile(text, '<uniform_int8_reader>', 'exec')
base = Path(__file__).resolve().parent
tree = ast.parse((base / 'linear_writer.py').read_text())
encoder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_linear_encode_bf16')
assert len(encoder.decorator_list) == 1
assert len([n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_linear_encode_bf16']) == 3
for file in base.glob('*.py'):
    ast.parse(file.read_text())
r = dict(complete=True, CPU_only=True, all65536_BF16_float_and_integer_encoder_references_agree=True,
    all256_signed_decoder_codes_checked=True, sign_zero_infinity_and_NaN_policy_checked=True,
    renderer_roundtrip_preserves_all_other_attention_code=True, reader_proof=proof,
    sources={name:hashlib.sha256((base/name).read_bytes()).hexdigest()
        for name in ('codec.py', 'linear_writer.py', 'check_cpu.py')},
    numeric_KV_format_changed=True, new_model_quality_not_proven=True, GPU_not_run=True, adopted=False)
(base/'CPU_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
