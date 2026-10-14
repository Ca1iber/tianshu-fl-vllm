# SPDX-License-Identifier: Apache-2.0
"""Add a 3D KV loop-unroll hint to the exact validated .cv source."""
import hashlib
import importlib.util
from pathlib import Path
import sys

CV_SHA = "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
SIGNATURE = '    KV_CACHE_MODIFIER: tl.constexpr = "",\n):'
NEW_SIGNATURE = '    KV_CACHE_MODIFIER: tl.constexpr = "",\n    KV_LOOP_UNROLL: tl.constexpr = 1,\n):'
LOOP = '    for j in range(loop_lo, loop_hi):'
NEW_LOOP = '    for j in tl.range(loop_lo, loop_hi, loop_unroll_factor=KV_LOOP_UNROLL if IS_3D else 1):'


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest() == CV_SHA
    text = raw.decode()
    assert text.count(SIGNATURE) == text.count(LOOP) == 1
    result = text.replace(SIGNATURE, NEW_SIGNATURE).replace(LOOP, NEW_LOOP)
    assert result.replace(NEW_SIGNATURE, SIGNATURE).replace(NEW_LOOP, LOOP).encode() == raw
    return result


def load_kernel(cv_source, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "generated_loop_unroll.py"
    source.write_text(render_source(Path(cv_source).read_bytes()))
    name = "_tianshu_cv_loop_unroll"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.kernel_unified_attention, dict(base_cv_sha256=CV_SHA,
        generated_source=str(source), generated_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        only_loop_header_and_constexpr_added=True, arithmetic_body_and_reducer_source_unchanged=True,
        installed_sources_modified=False, hint_does_not_prove_actual_lowered_unrolling=True)
