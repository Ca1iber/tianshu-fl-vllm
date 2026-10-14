# SPDX-License-Identifier: Apache-2.0
"""Permute physical CTA axes while preserving every logical Decode CTA."""
import hashlib
import importlib.util
from pathlib import Path
import sys

CV_SHA = "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
ORIGINAL = """    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0
"""


def assignments(axes):
    assert tuple(sorted(axes)) == (0, 1, 2)
    return (f"    q_block_global_idx = tl.program_id({axes[0]}) if IS_3D else tl.program_id(0)\n"
            f"    kv_head_idx = tl.program_id({axes[1]}) if IS_3D else tl.program_id(1)\n"
            f"    segm_idx = tl.program_id({axes[2]}) if IS_3D else 0\n")


def render_source(raw, axes):
    assert hashlib.sha256(raw).hexdigest() == CV_SHA
    text = raw.decode()
    assert text.count(ORIGINAL) == 1
    rendered = text.replace(ORIGINAL, assignments(axes))
    assert rendered.replace(assignments(axes), ORIGINAL).encode() == raw
    return rendered


def permute_grid(logical_grid, axes):
    assert len(logical_grid) == 3 and tuple(sorted(axes)) == (0, 1, 2)
    physical = [None] * 3
    for logical_axis, physical_axis in enumerate(axes):
        physical[physical_axis] = logical_grid[logical_axis]
    return tuple(physical)


def load_kernel(cv_source, directory, axes):
    raw = Path(cv_source).read_bytes()
    tag = "".join(map(str, axes))
    directory = Path(directory) / tag
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "generated_grid_permute.py"
    source.write_text(render_source(raw, axes))
    name = "tianshu_decode_grid_" + tag
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.kernel_unified_attention, dict(
        base_cv_sha256=CV_SHA, logical_to_physical_axes=list(axes),
        generated_source=str(source), generated_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        attention_arithmetic_changed=False, reducer_changed=False,
        two_dimensional_mapping_unchanged=True, installed_sources_modified=False)
