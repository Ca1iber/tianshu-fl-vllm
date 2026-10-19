# SPDX-License-Identifier: Apache-2.0
"""Execute actual view metadata conversion with packed/strided storage CPU fixtures."""
import ast
from pathlib import Path
from types import SimpleNamespace

base = Path(__file__).resolve().parent
node = next(n for n in ast.parse((base / 'value_planes.py').read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == 'value_plane')


class Storage:
    def __init__(self, size): self.size = size
    def data_ptr(self): return 1000000
    def nbytes(self): return self.size


class View:
    dtype = 'BF16'
    def set_(self, storage, offset, shape, stride):
        self.storage, self.offset, self.shape, self.steps = storage, offset, shape, stride
        return self
    def data_ptr(self): return self.storage.data_ptr() + 2 * self.offset
    def untyped_storage(self): return self.storage
    def stride(self): return self.steps


class Raw:
    dtype, ndim = 'U8', 4
    device = SimpleNamespace(type='cuda')
    def __init__(self, count, step, offset, short=False):
        self.shape, self.steps, self.offset = (count, 16, 2, 128), (step, 256, 128, 1), offset
        self.storage = Storage(offset + (count - 1) * step + 8192 - int(short))
    def stride(self, i=None): return self.steps if i is None else self.steps[i]
    def storage_offset(self): return self.offset
    def data_ptr(self): return self.storage.data_ptr() + self.offset
    def untyped_storage(self): return self.storage


torch = SimpleNamespace(uint8='U8', bfloat16='BF16', empty=lambda *a, **kw: View(),
    cuda=SimpleNamespace(is_current_stream_capturing=lambda: False))
environment = dict(torch=torch, _views={}, records={})
exec(compile(ast.Module(body=[node], type_ignores=[]), str(base / 'value_planes.py'), 'exec'), environment)
convert = environment['value_plane']
count = 0
for blocks in (1, 3, 17):
    for step in (12288, 12544, 24576):
        for offset in (4096, 4224):
            environment['_views'].clear()
            raw = Raw(blocks, step, offset)
            view = convert(raw)
            assert view.steps == (step // 2, 256, 128, 1)
            assert view.offset == offset // 2 and view.shape == raw.shape
            assert convert(raw) is view and convert(view) is view
            count += 1
for raw in (Raw(3, 8192, 4096), Raw(3, 12288, 4097), Raw(3, 12288, 4096, short=True)):
    environment['_views'].clear()
    try: convert(raw)
    except AssertionError: pass
    else: raise AssertionError('unsafe V metadata accepted')
assert count == 18
print(dict(actual_typed_view_cases=count, invalid_metadata_rejected=3, CPU_only=True))
