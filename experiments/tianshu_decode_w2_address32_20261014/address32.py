# SPDX-License-Identifier: Apache-2.0
"""Bounded address32 generated from the validated L0 source; no FP body edits."""
import hashlib
import importlib.util
from pathlib import Path
import re
import sys

L0_SHA = "4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e"


def bounds(tensors):
    values = [sum((n - 1) * abs(stride) for n, stride in zip(t.shape, t.stride()))
              + 2048 * sum(abs(s) for s in t.stride()) for t in tensors]
    assert all(0 <= x < 2**31 for x in values)
    return values


def load_variant(native, directory):
    helper = Path(__file__).resolve().parents[1] / "tianshu_native_prefill_20261012/zero_initialization.py"
    spec = importlib.util.spec_from_file_location("_tianshu_w2_address_l0_generator", helper)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    root = Path(directory)
    original, provenance = generator.load_kernel(native, root / "original_l0")
    raw = Path(provenance["generated_source"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == L0_SHA
    text, annotations = re.subn(r": tl\.int64\b", ": tl.int32", raw.decode())
    text, casts = re.subn(r"\.to\(\s*tl\.int64\s*\)", ".to(tl.int32)", text)
    assert annotations == 28 and casts == 6
    path = root / "l0_address32.py"
    path.write_text(text)
    name = "_tianshu_w2_l0_address32_" + hashlib.sha256(text.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return original, module, dict(original_l0_sha256=L0_SHA,
        generated_sha256=hashlib.sha256(text.encode()).hexdigest(), annotations_changed=annotations,
        address_casts_changed=casts, integer_element_offsets_bits=32, physical_pointer_types_unchanged=True,
        floating_point_source_changes=False, zero_initialization=provenance)
