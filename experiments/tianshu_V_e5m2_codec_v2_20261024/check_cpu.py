# SPDX-License-Identifier: Apache-2.0
"""Exhaustive bit-domain and source-scope proof; standard library only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import codec

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
finite = nonfinite = saturated = 0
max_normal_relative = 0.0
for bits in range(65536):
    value = codec.bf16_scalar(bits)
    actual = codec.encode_bf16(bits)
    expected = codec.reference_encode(value)
    assert actual == expected, (bits, value, actual, expected)
    assert actual & 127 <= 123
    if math.isfinite(value):
        finite += 1
        if abs(value) > 57344:
            saturated += 1
        if 2.0 ** -14 <= abs(value) <= 57344:
            decoded = codec.reference_decode(actual)
            max_normal_relative = max(max_normal_relative, abs(decoded - value) / abs(value))
    else:
        nonfinite += 1
assert finite == 65280 and nonfinite == 256
valid = reserved = 0
for code in range(256):
    actual = struct.unpack('<e', struct.pack('<H', code << 8))[0]
    expected = codec.reference_decode(code)
    if math.isfinite(expected):
        valid += 1
        assert actual == expected
        assert math.copysign(1.0, actual) == math.copysign(1.0, expected)
        bits32 = struct.unpack('>I', struct.pack('>f', actual))[0]
        assert bits32 & 65535 == 0, 'finite E5M2 point must fit BF16 exactly'
        assert codec.encode_bf16(bits32 >> 16) == code
    else:
        reserved += 1
        assert math.isnan(actual) if math.isnan(expected) else actual == expected
assert valid == 248 and reserved == 8
proof = {}
for mode in codec.MODES:
    source, record = codec.render_reader(codec.READER.read_bytes(), mode)
    (a.output / f'reader_{mode}.py').write_text(source)
    compile(source, f'reader_{mode}.py', 'exec')
    proof[mode] = record
source, proof['writer'] = codec.render_writer(codec.WRITER.read_bytes())
(a.output / 'writer.py').write_text(source)
compile(source, 'writer.py', 'exec')
assert 'torch' not in sys.modules and 'triton' not in sys.modules
sources = [Path(__file__).resolve(), Path(codec.__file__).resolve(),
           codec.BASE / 'PARENT_SOURCES_32g.sha256', codec.READER, codec.WRITER]
gate = dict(complete=True, CPU_only=True, generated_UTC=datetime.now(timezone.utc).isoformat(),
    finite_BF16_checked=finite, nonfinite_BF16_checked=nonfinite,
    independently_rounded_all_BF16_encodings=True,
    decoded_codes=256, finite_codes=valid, reserved_codes=reserved,
    finite_codes_exact_in_FP16_and_BF16_and_roundtrip=True,
    finite_input_saturation_count=saturated,
    normal_unsaturated_max_relative_error=max_normal_relative,
    source_scope_proof=proof,
    sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
    physical_K8V16_page_bytes=12288, proposed_K8V8_page_bytes=8192,
    no_Torch_Triton_import=True, GPU_numerical_qualified=False,
    model_quality_qualified=False, service_tested=False, adopted=False)
(a.output / 'CPU_GATE_32g.json').write_text(json.dumps(gate, indent=2) + '\n')
print(json.dumps({k:gate[k] for k in ('complete','finite_BF16_checked','nonfinite_BF16_checked','finite_codes','reserved_codes','normal_unsaturated_max_relative_error')}))
