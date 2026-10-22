#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
group_runner=/workspace/benchmark-results/20261022_down_group_v3_runner_tianshu
mkdir "$group_runner"
TRITON_CACHE_DIR=/workspace/benchmark-results/20261022_prefill_control_binary_all_GPU_tianshu/qualified_model_cache python3 -u experiments/tianshu_down_group_gpu_v3_20261022/probe_gpu.py --output /workspace/benchmark-results/20261022_down_group_v3_GPU_tianshu --stop-at-utc 2026-10-22T15:58:00+00:00 > "$group_runner/GPU.log" 2>&1
ixsmi > "$group_runner/hardware_after_GPU.txt"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -u experiments/tianshu_down_group_gpu_v3_20261022/verify_cpu.py > "$group_runner/CPU.log" 2>&1
