#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
down42_runner=/workspace/benchmark-results/20261022_native_int8_down42_runner_tianshu
export TRITON_CACHE_DIR="$down42_runner/qualified_cache"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_native_int8_down42_20261022/probe_gpu.py --output /workspace/benchmark-results/20261022_native_int8_down42_GPU_tianshu --stop-at-utc 2026-10-22T15:58:00+00:00 > "$down42_runner/gpu.log" 2>&1 &
printf '%s\n' "$!" > "$down42_runner/gpu_pid"
wait "$(cat "$down42_runner/gpu_pid")"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_native_int8_down42_20261022/verify_cpu.py --input /workspace/benchmark-results/20261022_native_int8_down42_GPU_tianshu > "$down42_runner/cpu_audit.log" 2>&1
