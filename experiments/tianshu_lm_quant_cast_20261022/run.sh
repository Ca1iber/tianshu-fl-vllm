#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
sha256sum --check --status experiments/tianshu_lm_quant_cast_20261022/EXECUTION.sha256
cast_runner=/workspace/benchmark-results/20261022_lm_quant_cast_runner_tianshu
TRITON_CACHE_DIR="$cast_runner/cache" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_lm_quant_cast_20261022/probe_gpu.py --output /workspace/benchmark-results/20261022_lm_quant_cast_GPU_tianshu --stop-at-utc 2026-10-22T15:58:00+00:00 > "$cast_runner/GPU.log" 2>&1
ixsmi > "$cast_runner/hardware_after_GPU.txt"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_lm_quant_cast_20261022/verify_cpu.py > "$cast_runner/CPU.log" 2>&1
