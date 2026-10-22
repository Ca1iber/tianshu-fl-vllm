#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
w8a16_runner=/workspace/benchmark-results/20261022_down_W8A16_v2_runner_tianshu
mkdir "$w8a16_runner"
TRITON_CACHE_DIR=/workspace/benchmark-results/20261022_prefill_control_binary_all_GPU_tianshu/qualified_model_cache python3 -u experiments/tianshu_down_W8A16_gpu_v2_20261022/probe_gpu.py > "$w8a16_runner/GPU.log" 2>&1
ixsmi > "$w8a16_runner/hardware_after_GPU.txt"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -u experiments/tianshu_down_W8A16_gpu_v2_20261022/verify_cpu.py > "$w8a16_runner/CPU.log" 2>&1
