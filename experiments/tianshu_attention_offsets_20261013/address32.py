# SPDX-License-Identifier: Apache-2.0
"""Generate process-only native attention with bounded 32-bit element offsets."""
import hashlib
import importlib.util
from pathlib import Path
import re
import sys


SOURCE_SHA = 'bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0'


def offset_bound(tensor):
    # Add a conservative padded-tile margin to cover masked lanes too.
    strides = [abs(value) for value in tensor.stride()]
    return sum((size - 1) * stride for size, stride in zip(tensor.shape, strides)) + 2048 * sum(strides)


def assert_safe(tensors):
    bounds = [offset_bound(tensor) for tensor in tensors]
    assert all(0 <= value < 2**31 for value in bounds)
    return bounds


def load_variant(native, directory):
    raw = Path(native.__file__).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    text = raw.decode()
    text, annotations = re.subn(r': tl\.int64\b', ': tl.int32', text)
    text, casts = re.subn(r'\.to\(\s*tl\.int64\s*\)', '.to(tl.int32)', text)
    assert annotations and casts
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'native_address32.py'
    path.write_text(text)
    name = '_tianshu_native_address32_' + hashlib.sha256(text.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, dict(original_source_sha256=SOURCE_SHA, generated_source=str(path),
                        generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        address_stride_annotations_changed=annotations, offset_casts_changed=casts,
                        floating_point_body_unmodified=True, pointer_types_unmodified=True)


class Stage1Kernel:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            if kwargs.get('IS_3D'):
                kwargs['num_stages'] = 1
                kwargs['num_warps'] = 4
            return self.kernel[grid](*args, **kwargs)
        return invoke
