# SPDX-License-Identifier: Apache-2.0
"""CPU oracle for earliest, longest suffix matches; not a GPU proof."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import random


def reference(tokens, enabled=True):
    if enabled:
        for ngram in (5, 4, 3):
            if len(tokens) <= ngram:
                continue
            suffix = tokens[-ngram:]
            for first in range(len(tokens) - ngram):
                if tokens[first:first + ngram] == suffix:
                    token = tokens[first + ngram]
                    return token, int(token != -1)
    return -1, 0


def scored(tokens, enabled=True, cap=131072):
    invalid = 3 * (cap + 1)
    best = invalid
    if enabled and len(tokens) >= 4:
        for end in range(2, len(tokens) - 1):
            if tokens[end-2:end+1] != tokens[-3:]:
                continue
            matched = 3
            if end >= 3 and tokens[end-3] == tokens[-4]:
                matched = 4
                if end >= 4 and len(tokens) >= 5 and tokens[end-4] == tokens[-5]:
                    matched = 5
            best = min(best, (5-matched) * (cap+1) + end-matched+1)
    if best == invalid:
        return -1, 0
    matched = 5-best//(cap+1)
    first = best % (cap+1)
    token = tokens[first+matched]
    return token, int(token != -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checks = 0
    for size in range(15):
        for sequence in product((0, 1), repeat=size):
            tokens = list(sequence)
            assert scored(tokens) == reference(tokens), tokens
            checks += 1
    rng = random.Random(20261019031)
    for size in (3, 4, 5, 6, 16, 127, 4096, 5120, 8192, 131072):
        rows = [[3] * size, [i % 7 for i in range(size)],
                [rng.randrange(-1, 130560) for _ in range(size)]]
        if size > 12:
            row = list(rows[-1]); row[-5:] = row[3:8]; rows.append(row)
        for tokens in rows:
            for enabled in (False, True):
                assert scored(tokens, enabled) == reference(tokens, enabled)
                checks += 1
    # Address bound includes the last valid element, not just a byte count.
    assert 255*131072+131071 < 2**31
    assert 3*(131072+1) < 2**31
    report = dict(complete=True, CPU_only=True, exact_cases=checks,
        longest_5_4_3_then_earliest_start_verified=True,
        disabled_and_short_rows_verified=True, negative_sentinel_verified=True,
        maximum_context=131072, maximum_row_address=255*131072+131071,
        score_sentinel=3*(131072+1), GPU_test=False, adopted=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_source_sha256=hashlib.sha256(Path(__file__).with_name('search.py').read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
