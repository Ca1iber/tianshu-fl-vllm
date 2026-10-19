# SPDX-License-Identifier: Apache-2.0
"""Bounded element-offset arithmetic; physical pointers and KV values stay unchanged."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import re
import sys

SOURCE_SHA = 'dc69f525d8c7686a8977f50e901af39820ce92486894890631c53fc01dc46700'


def offset_bound(tensor):
    shape, strides = tuple(tensor.shape), tuple(tensor.stride())
    assert len(shape) == len(strides) and all(n > 0 for n in shape)
    assert all(s >= 0 for s in strides)
    # Include masked lanes outside each logical extent, as in the earlier native probe.
    return sum((n - 1) * s for n, s in zip(shape, strides)) + 2048 * sum(strides)


def assert_safe(tensors):
    bounds = [offset_bound(tensor) for tensor in tensors]
    assert bounds and all(0 <= value < 2**31 for value in bounds)
    return bounds


def render(raw):
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    text, annotations = re.subn(r': tl\.int64\b', ': tl.int32', raw.decode())
    text, casts = re.subn(r'\.to\(\s*tl\.int64\s*\)', '.to(tl.int32)', text)
    assert (annotations, casts) == (28, 6)

    class Expected(ast.NodeTransformer):
        def __init__(self):
            self.count = 0

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'tl' and node.attr == 'int64':
                self.count += 1
                node.attr = 'int32'
            return self.generic_visit(node)

    expected = Expected()
    tree = expected.visit(ast.parse(raw))
    assert expected.count == 34
    assert ast.dump(tree, include_attributes=False) == ast.dump(ast.parse(text), include_attributes=False)
    return text, dict(original_source_sha256=SOURCE_SHA,
        generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        address_stride_annotations_changed=annotations, offset_casts_changed=casts,
        AST_only_integer_type_changes=True, floating_point_body_unmodified=True,
        pointer_types_unmodified=True, writer_and_codec_values_unmodified=True)


def load_variant(reader, directory):
    text, proof = render(Path(reader.__file__).read_bytes())
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'generated_address32.py'
    path.write_text(text)
    name = '_tianshu_k8v16_address32_' + proof['generated_sha256'][:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, dict(proof, generated_source=str(path))
