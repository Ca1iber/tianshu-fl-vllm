# SPDX-License-Identifier: Apache-2.0
"""CPU proofs of exact packed local ranks and adjacent radix digit detection."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random


def check_packed(labels):
    assert 0 < len(labels) < 65536
    total = 0
    counts = [0, 0]
    for label in labels:
        weight = 1 if label == 0 else 1 << 16 if label == 1 else 0
        total += weight
        assert 0 <= total <= 0xFFFFFFFF
        # Independent scalar counts before this element define its rank.
        ranks = total - weight
        assert (ranks & 65535, ranks >> 16) == tuple(counts)
        if label < 2:
            counts[label] += 1
    assert (total & 65535, total >> 16) == tuple(counts)


def check_detection(keys, bits, digit_bits):
    assert bits in (32, 64) and 32 % digit_bits == 0
    mask = (1 << digit_bits) - 1
    mismatch = 0
    for key in keys:
        assert 0 <= key < 1 << bits
        mismatch |= key ^ (key >> digit_bits)
    words = [mismatch & 0xFFFFFFFF, mismatch >> 32]
    checked = 0
    for p in range(1, bits // digit_bits):
        offset = (p - 1) * digit_bits
        predicted = ((words[offset // 32] >> (offset % 32)) & mask) == 0
        actual = all(((key >> (p * digit_bits)) & mask) == ((key >> offset) & mask) for key in keys)
        assert predicted == actual
        checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(20261014)
    packed_cases = 0
    for labels in itertools.product(range(3), repeat=8):
        check_packed(labels)
        packed_cases += 1
    for n in (1, 7, 2048, 65535):
        for labels in ([0]*n, [1]*n, [2]*n, [i % 3 for i in range(n)], [rng.randrange(3) for _ in range(n)]):
            check_packed(labels)
            packed_cases += 1
    cases = fields = 0
    for bits in (32, 64):
        for digit_bits in (1, 4, 8):
            for n in (1, 2, 7, 31):
                patterns = [[0]*n, [(1 << bits)-1]*n, [rng.getrandbits(bits) for _ in range(n)],
                            [rng.getrandbits(bits) & ~65535 for _ in range(n)], [int('a'*(bits//4),16)]*n]
                for keys in patterns:
                    fields += check_detection(keys,bits,digit_bits)
                    cases += 1
    # Equal histograms alone do not prove an identity pass.
    keys = [0x10, 0x01]
    assert sorted(key & 15 for key in keys) == sorted((key >> 4) & 15 for key in keys)
    assert not all((key & 15) == ((key >> 4) & 15) for key in keys)
    fields += check_detection(keys,32,4)
    cases += 1
    result = dict(complete=True, checked_utc=datetime.now(timezone.utc).isoformat(),
        cpu_integer_equivalence_only=True, GPU_lowering_and_performance_not_tested=True,
        packed_cases=packed_cases, maximum_tested_local_tile=65535,
        exhaustive_three_state_tile8=True, independent_rank_and_count_reference=True,
        digit_cases=cases, digit_fields=fields, histogram_only_counterexample_rejected=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
