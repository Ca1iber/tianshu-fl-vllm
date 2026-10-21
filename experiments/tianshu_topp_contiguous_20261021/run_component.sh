#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
export VLLM_PLUGINS=fl OMP_NUM_THREADS=8
exec python3 -u experiments/tianshu_topp_contiguous_20261021/probe_gpu.py --output /workspace/benchmark-results/20261021_topp_contiguous_GPU_tianshu --stop-at-utc 2026-10-21T15:58:00+00:00
