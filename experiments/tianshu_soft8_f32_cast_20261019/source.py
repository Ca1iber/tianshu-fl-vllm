# SPDX-License-Identifier: Apache-2.0
"""Keep every encoded value; try a binary32 intermediate for cache decoding."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

SOURCE_PARENT = Path(__file__).resolve().parents[1] / 'tianshu_soft8_engine_v2_20261019/source.py'
spec = importlib.util.spec_from_file_location('_soft8_f32_parent_source', SOURCE_PARENT)
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
prior = parent.prior
encode_bits, encode_unsigned = parent.encode_bits, parent.encode_unsigned
decode_scalar, render_writer = parent.decode_scalar, parent.render_writer
LEGACY, VARIANT, WRITER_SHA = parent.LEGACY, parent.VARIANT, parent.WRITER_SHA

OLD = '''        bits = tl.where(magnitude >= 8, (magnitude << 4) + 14848, 0)
        bits = bits | ((packed & 128) << 8)
        return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(Q.dtype)'''
NEW = '''        bits = tl.where(magnitude >= 8, (magnitude << 20) + 973078528, 0)
        bits = bits | ((packed & 128) << 24)
        return bits.to(tl.float32, bitcast=True).to(Q.dtype)'''


def render_reader(raw):
    original, proof = prior.render_source(raw)
    assert original.count(OLD) == 1
    text = original.replace(OLD, NEW)
    assert text.replace(NEW, OLD) == original
    for name in ('kernel_unified_attention', 'reduce_segments'):
        def function(s):
            return next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(function(text)) == ast.dump(function(original))
    return text, dict(proof, parent_generated_sha256=proof['generated_sha256'],
        generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        codec_intermediate='binary32_bitcast_then_BF16', encoded_format_unchanged=True)


def load_modules(cv_source, writer_source, root):
    reader, writer, proof = parent.load_modules(cv_source, writer_source, root)
    text, f32_proof = render_reader(Path(cv_source).read_bytes())
    output = Path(root) / 'float_reader' / 'generated_soft8_f32.py'
    output.parent.mkdir()
    output.write_text(text)
    name = '_tianshu_soft8_f32_reader'
    spec = importlib.util.spec_from_file_location(name, output)
    f32 = importlib.util.module_from_spec(spec)
    sys.modules[name] = f32
    spec.loader.exec_module(f32)
    reader.f32_variant = f32
    proof['f32_reader'] = dict(f32_proof, generated_source=str(output))
    return reader, writer, proof
