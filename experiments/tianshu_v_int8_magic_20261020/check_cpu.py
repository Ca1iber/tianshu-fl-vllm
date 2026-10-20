# SPDX-License-Identifier: Apache-2.0
"""CPU contract and exact source edit checks; run on the Tianshu container."""
import argparse
import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import codec


def rational_encode(value, denominator):
    if math.isnan(value):
        return 0
    if math.isinf(value):
        return 128 if value < 0 else 127
    exact = Fraction.from_float(value) * denominator
    lower, remainder = divmod(exact.numerator, exact.denominator)
    twice = 2 * remainder
    upper = twice > exact.denominator or (twice == exact.denominator and lower % 2 != 0)
    signed = lower + int(upper)
    signed = -128 if signed < -128 else 127 if signed > 127 else signed
    return signed % 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader-source', type=Path, required=True,
                        help='Frozen generated_cache_policy.py, SHA checked by the prior codec')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    writer = base.parent / 'tianshu_soft8_mixed_planes_20261019/writer.py'
    result = dict(complete=False, scope='CPU numeric contract and reversible source edits only',
                  GPU_executed=False, service_executed=False, adoption_allowed=False, cases=[])
    for denominator in codec.DENOMINATORS:
        finite_count = 0
        for bits in range(65536):
            value = codec.bf16_scalar(bits)
            actual = codec.encode_scalar(value, denominator)
            assert actual == rational_encode(value, denominator), (denominator, bits, actual)
            if math.isfinite(value):
                finite_count += 1
                if -128 / denominator <= value <= 127 / denominator:
                    decoded = codec.decode_scalar(actual, denominator)
                    assert abs(decoded - value) <= .5 / denominator
        for code in range(256):
            decoded = codec.decode_scalar(code, denominator)
            assert codec.encode_scalar(decoded, denominator) == code
            assert decoded == (code if code < 128 else code - 256) / denominator
        boundary_count = 0
        for integer in range(-130, 131):
            for fraction in (Fraction(1, 2) - Fraction(1, 1024), Fraction(1, 2),
                             Fraction(1, 2) + Fraction(1, 1024)):
                value = float((integer + fraction) / denominator)
                assert codec.encode_scalar(value, denominator) == rational_encode(value, denominator)
                boundary_count += 1
        reader_text, reader_proof = codec.render_reader(args.reader_source.read_bytes(), denominator)
        writer_text, writer_proof = codec.render_writer(writer.read_bytes(), denominator)
        ast.parse(reader_text)
        ast.parse(writer_text)
        result['cases'].append(dict(denominator=denominator, scale=1 / denominator,
            BF16_bit_patterns=65536, finite_BF16_patterns=finite_count,
            independent_exact_rational_reference=True, decoded_code_points=256,
            rounding_boundary_cases=boundary_count, reader=reader_proof, writer=writer_proof,
            generated_Python_AST_valid=True, Triton_compilation_tested=False))
    for invalid in (0, -1, 3, 64, 16.0, True):
        try:
            codec.validate_denominator(invalid)
        except ValueError:
            continue
        raise AssertionError(('accepted invalid denominator', invalid))
    result['complete'] = True
    result['frozen_sources'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), Path(codec.__file__), writer, args.reader_source, codec.PARENT)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()