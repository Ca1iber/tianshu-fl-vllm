# SPDX-License-Identifier: Apache-2.0
"""Generate bounded int32 element offsets from the actual .cv kernel source."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import re
import sys

CV_SHA = "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"


def bounds(tensors):
    values = [sum((n - 1) * abs(s) for n, s in zip(t.shape, t.stride()))
              + 2048 * sum(abs(s) for s in t.stride()) for t in tensors]
    assert values and all(0 <= x < 2**31 for x in values)
    return values


def normalized_ast(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "tl" and node.attr in ("int32", "int64"):
                node.attr = "INTEGER_ADDRESS"
    return ast.dump(tree, include_attributes=False)


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest() == CV_SHA
    text, annotations = re.subn(r": tl\.int64\b", ": tl.int32", raw.decode())
    text, casts = re.subn(r"\.to\(\s*tl\.int64\s*\)", ".to(tl.int32)", text)
    assert annotations == 28 and casts == 6
    assert normalized_ast(text) == normalized_ast(raw.decode())
    return text


def load_variant(cv_source, directory):
    text = render_source(Path(cv_source).read_bytes())
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "cv_address32.py"
    source.write_text(text)
    name = "_tianshu_cv_address32_" + hashlib.sha256(text.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, dict(base_cv_sha256=CV_SHA, generated_source=str(source),
        generated_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        annotations_changed=28, casts_changed=6, only_integer_address_types_changed=True,
        pointer_types_unchanged=True, floating_point_AST_unchanged=True,
        installed_sources_modified=False)


class ReducerRecorder:
    def __init__(self, original, on_compile=None):
        self.original = original
        self.metadata = {}
        self.on_compile = on_compile

    def __getitem__(self, grid):
        def launch(**kwargs):
            assert "num_warps" not in kwargs and "num_stages" not in kwargs
            compiled = self.original[grid](**kwargs)
            self.metadata = dict(source_module=self.original.fn.__module__,
                actual_num_warps=compiled.metadata.num_warps,
                actual_num_stages=compiled.metadata.num_stages, grid=list(grid),
                TILE_SIZE=kwargs["TILE_SIZE"], NUM_SEGMENTS_PER_SEQ=kwargs["NUM_SEGMENTS_PER_SEQ"],
                n_regs=getattr(compiled, "n_regs", None), n_spills=getattr(compiled, "n_spills", None))
            if self.on_compile is not None: self.on_compile(compiled, dict(self.metadata))
            return compiled
        return launch
