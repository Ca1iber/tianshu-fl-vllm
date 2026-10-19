# SPDX-License-Identifier: Apache-2.0
"""Reuse the measured GEO arithmetic verbatim; alter only its launch grid."""
import hashlib
import importlib.util
from pathlib import Path
import sys

SOURCE_SHA = '64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
GEO_SHA = '09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
parent = Path(__file__).resolve().parents[1]/'tianshu_topp_geometric_20261016/precision.py'
spec = importlib.util.spec_from_file_location('_grid_geometric_base', parent)
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
topk_ast = previous.topk_ast


def render_source(raw, mode):
    assert mode == 'geo'
    source = previous.render_source(raw, 'geo')
    assert hashlib.sha256(source.encode()).hexdigest() == GEO_SHA
    return source


def load_kernel(native_source, directory):
    raw = Path(native_source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'generated_geo.py'; path.write_text(render_source(raw, 'geo'))
    name = '_tianshu_topp_grid_geo'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return {'geo': module._topk_topp_kernel}, dict(native_source_sha256=SOURCE_SHA,
        base_geo_sha256=GEO_SHA, modes={'geo': dict(source=str(path), sha256=GEO_SHA)},
        strict_CPU_tolerance_unchanged=1e-6, p_unchanged=True,
        RNG_implementation_unchanged=True, all_GPU_arithmetic_FP32=True,
        standalone_search_limit_unchanged=32, standalone_interval_tolerance_unchanged=1e-12,
        native_TopK_AST_unchanged=True, installed_source_modified=False,
        GEO_arithmetic_source_unchanged=True)
