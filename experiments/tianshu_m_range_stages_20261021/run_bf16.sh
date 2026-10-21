#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
for pid in 249716 255255; do
    state=$(ps -p "$pid" -o stat= || true)
    if test -n "$state" && test "${state:0:1}" != Z; then
        printf 'Current formal validation PID %s still active; not starting GPU component.\n' "$pid" >&2
        exit 3
    fi
done
exec env VLLM_PLUGINS=fl python3 experiments/tianshu_m_range_stages_20261021/probe_bf16.py \
    --output /workspace/benchmark-results/20261021_m_range_BF16_GPU_tianshu \
    --stop-at-utc 2026-10-21T15:58:00+00:00
