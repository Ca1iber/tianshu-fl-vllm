#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
exec python3 -u experiments/tianshu_multiquery_splitkv_20261021/probe_gpu.py --output /workspace/benchmark-results/20261021_multiquery_splitkv_GPU_tianshu --stop-at-utc 2026-10-21T15:58:00+00:00
