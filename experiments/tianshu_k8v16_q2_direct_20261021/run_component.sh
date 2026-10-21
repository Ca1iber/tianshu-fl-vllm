#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
for pid in 247467 249716; do
    state=$(ps -p "$pid" -o stat= || true)
    if test -n "$state" && test "${state:0:1}" != Z; then
        printf 'Validation PID %s is still active; no GPU component started.\n' "$pid" >&2
        exit 3
    fi
done
exec env VLLM_PLUGINS=fl python3 experiments/tianshu_k8v16_q2_direct_20261021/probe_gpu.py \
    --output /workspace/benchmark-results/20261021_k8v16_q2_direct_GPU_tianshu \
    --stop-at-utc 2026-10-21T15:58:00+00:00
