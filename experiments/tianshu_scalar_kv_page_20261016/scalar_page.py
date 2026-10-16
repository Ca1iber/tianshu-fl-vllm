# SPDX-License-Identifier: Apache-2.0
"""Single page-table load when a decode KV tile is exactly one page."""
import hashlib
import importlib.util
from pathlib import Path
import sys

CV_SHA = '8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289'
SIGNATURE = '    KV_CACHE_MODIFIER: tl.constexpr = "",\n):'
NEW_SIGNATURE = '    KV_CACHE_MODIFIER: tl.constexpr = "",\n    SCALAR_KV_PAGE: tl.constexpr = False,\n):'
LOAD = '''        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)'''
NEW_LOAD = '''        if SCALAR_KV_PAGE and IS_3D and TILE_SIZE == BLOCK_SIZE:
            page_id = tl.load(block_tables_ptr + block_table_offset + j).to(tl.int64)
            physical_block_idx = page_id + tl.zeros((TILE_SIZE,), dtype=tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)'''


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest() == CV_SHA
    text = raw.decode()
    assert text.count(SIGNATURE) == text.count(LOAD) == 1
    changed = text.replace(SIGNATURE, NEW_SIGNATURE).replace(LOAD, NEW_LOAD)
    assert changed.replace(NEW_SIGNATURE, SIGNATURE).replace(NEW_LOAD, LOAD).encode() == raw
    return changed


def load_kernel(cv_source, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'generated_scalar_page.py'
    source.write_text(render_source(Path(cv_source).read_bytes()))
    name = '_tianshu_scalar_kv_page'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.kernel_unified_attention, dict(base_cv_sha256=CV_SHA,
        generated_source=str(source), generated_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        only_page_load_and_constexpr_changed=True, floating_arithmetic_and_reducer_source_unchanged=True,
        integer_widths_unchanged=True, guarded_by_IS_3D_and_equal_page_tile=True,
        installed_sources_modified=False)
