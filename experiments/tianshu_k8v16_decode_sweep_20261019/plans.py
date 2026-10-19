# SPDX-License-Identifier: Apache-2.0
"""K8V16 may have another launch optimum; keep the measured policy as mandatory control."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'tianshu_k8v16_pipeline_20261019/dispatch_policy.py'
spec = importlib.util.spec_from_file_location('_k8v16_measured_decode_policy', path)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
PLANS = {
    'current': None,
    't16w1s2': (16, 1, 2, 1),
    't16w1s8': (16, 1, 8, 1),
    't16w1s16': (16, 1, 16, 1),
    't16w2s4': (16, 2, 4, 1),
    't32w1s4': (32, 1, 4, 1),
    't32w2s4': (32, 2, 4, 1),
    't32w2s8': (32, 2, 8, 1),
    't64w2s4': (64, 2, 4, 1),
    't64w4s4': (64, 4, 4, 1),
    't32w2s4_stage2': (32, 2, 4, 2),
}
MODES = tuple(PLANS)


def configuration(name, batch, context):
    assert 1 <= batch <= 64 and context > 0
    values = PLANS[name]
    if values is None:
        return dict(policy.choose_config(batch, context))
    tile, warps, segments, stages = values
    return dict(TILE_SIZE=tile, num_warps=warps, segments=segments, num_stages=stages)
