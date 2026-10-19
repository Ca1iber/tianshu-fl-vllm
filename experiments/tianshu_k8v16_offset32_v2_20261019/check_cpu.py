# SPDX-License-Identifier: Apache-2.0
"""Check the exact measured reader and actual K8V16 cache-view metadata on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
from source import assert_safe, render

p = argparse.ArgumentParser()
p.add_argument('--reader', type=Path, required=True)
p.add_argument('--engine-record', type=Path, required=True)
a = p.parse_args()
text, proof = render(a.reader.read_bytes())
compile(text, '<generated_address32>', 'exec')
engine = json.loads(a.engine_record.read_text())
assert engine['layout'] == 'K8V16'


class TensorMetadata:
    def __init__(self, shape, strides):
        self.shape, self.strides = shape, strides

    def stride(self):
        return self.strides


cases = []
for view in engine['cache_views'][-1]:
    n = view['shape'][0]
    cases += [TensorMetadata((n, 16, 2, 128), (view['strides'][0], 256, 128, 1)),
        TensorMetadata((n, 16, 2, 128), tuple(view['actual_V_strides']))]
assert len(cases) == 84
bounds = assert_safe(cases)
# Additional strided fixtures and realistic metadata/scratch spans, not GPU results.
for batch in (1, 8, 16, 24, 32, 64, 128, 256):
    cases = [TensorMetadata((batch, 16, 128), (2560, 128, 1)),
        TensorMetadata((batch, 16, 64, 128), (16*64*128, 64*128, 128, 1)),
        TensorMetadata((batch, 8192), (8192, 1)),
        TensorMetadata((batch+1,), (1,))]
    bounds += assert_safe(cases)
rejected = 0
for tensor in (TensorMetadata((200000, 16, 2, 128), (12288, 256, 128, 1)),
    TensorMetadata((1,), (2**31,)), TensorMetadata((2,), (-1,))):
    try:
        assert_safe([tensor])
    except AssertionError:
        rejected += 1
    else:
        raise AssertionError('unsafe metadata accepted')
assert rejected == 3
base = Path(__file__).resolve().parent
result = dict(complete=True, CPU_only=True, reader_proof=proof,
    actual_K8V16_cache_views_checked=84, metadata_and_scratch_fixtures=32,
    maximum_checked_element_offset=max(bounds), rejected_unsafe_metadata=3,
    input_engine_record_sha256=hashlib.sha256(a.engine_record.read_bytes()).hexdigest(),
    sources={name:hashlib.sha256((base/name).read_bytes()).hexdigest()
        for name in ('source.py', 'check_cpu.py')},
    GPU_compile_and_numerics_not_run=True, service_not_run=True, adopted=False)
(base/'CPU_GATE_32g.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
