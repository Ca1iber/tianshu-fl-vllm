# SPDX-License-Identifier: Apache-2.0
"""Reuse the qualified K decoder; select the actually measured typed-plane writer."""
import hashlib
import importlib.util
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent / 'tianshu_soft8_engine_v2_20261019/source.py'
spec = importlib.util.spec_from_file_location('_k8v16_qualified_codec_parent', PARENT)
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
LEGACY, VARIANT, WRITER_SHA = parent.LEGACY, parent.VARIANT, parent.WRITER_SHA
encode_bits, encode_unsigned, decode_scalar = parent.encode_bits, parent.encode_unsigned, parent.decode_scalar
prior, render_writer = parent.prior, parent.render_writer


def load_modules(cv_source, writer_source, root):
    reader, unused, proof = parent.load_modules(cv_source, writer_source, root)
    import writer_bridge
    proof['actual_writer_bridge'] = str(Path(writer_bridge.__file__))
    proof['actual_writer_source'] = str(writer_bridge.WRITER)
    proof['actual_writer_sha256'] = hashlib.sha256(writer_bridge.WRITER.read_bytes()).hexdigest()
    proof['stored_K_dtype'] = 'torch.uint8'
    proof['stored_V_dtype'] = 'torch.bfloat16'
    proof['physical_page_bytes'] = 12288
    proof['original_generated_writer_not_used_for_K8V16'] = True
    return reader, writer_bridge, proof
