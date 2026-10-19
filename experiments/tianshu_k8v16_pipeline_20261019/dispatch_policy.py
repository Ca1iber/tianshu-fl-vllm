# SPDX-License-Identifier: Apache-2.0
"""Fixed software8 policy selected from the independent decode scan."""


def choose_config(n, max_length):
    assert n > 0 and max_length > 0
    if n > 64:
        return None
    segments = (4 if max_length <= 128 else 64 if n == 1 else 32 if n <= 3
                else 16 if n <= 8 else 8 if n <= 16 else 16 if n <= 24 else 4)
    return dict(segments=segments, TILE_SIZE=16, num_warps=1, num_stages=1)
