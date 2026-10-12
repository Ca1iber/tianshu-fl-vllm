# SPDX-License-Identifier: Apache-2.0
"""Generate a checked native copy with an explicit zero no-sink denominator."""
import hashlib
import importlib.util
from pathlib import Path
import sys

SOURCE_SHA = 'bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0'


def load_kernel(native, directory):
    raw = Path(native.__file__).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    text = raw.decode()
    old = '    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)'
    assert text.count(old) == 1
    new = '''    if USE_SINKS:
        L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    else:
        L = tl.zeros([BLOCK_M], dtype=tl.float32)'''
    text = text.replace(old, new)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'original_native.py').write_bytes(raw)
    output = root / 'generated_zero_initialization.py'
    output.write_text(text)
    name = 'tianshu_native_zero_initialization'
    spec = importlib.util.spec_from_file_location(name, output)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.kernel_unified_attention, {
        'original_source_sha256': SOURCE_SHA,
        'generated_source': str(output),
        'generated_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'no_sink_denominator_initial_value': 0,
        'sink_initialization_unchanged': True,
        'installed_sources_modified': False}
