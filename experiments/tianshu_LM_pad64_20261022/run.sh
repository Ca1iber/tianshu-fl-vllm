#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
sha256sum --check --status experiments/tianshu_LM_pad64_20261022/EXECUTION.sha256
pad_runner=/workspace/benchmark-results/20261022_LM_pad64_runner_tianshu
TRITON_CACHE_DIR="$pad_runner/cache" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_LM_pad64_20261022/probe_gpu.py --output /workspace/benchmark-results/20261022_LM_pad64_GPU_tianshu --stop-at-utc 2026-10-22T15:58:00+00:00 > "$pad_runner/GPU.log" 2>&1
ixsmi > "$pad_runner/hardware_after_GPU.txt"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_LM_pad64_20261022/verify_cpu.py > "$pad_runner/CPU.log" 2>&1
