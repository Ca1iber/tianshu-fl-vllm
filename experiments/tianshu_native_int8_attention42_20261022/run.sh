#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
projection_runner=/workspace/benchmark-results/20261022_native_int8_attention42_runner_tianshu
mkdir "$projection_runner"
TRITON_CACHE_DIR=/workspace/benchmark-results/20261022_prefill_control_binary_all_GPU_tianshu/qualified_model_cache python3 -u experiments/tianshu_native_int8_attention42_20261022/probe_gpu.py > "$projection_runner/GPU.log" 2>&1
ixsmi > "$projection_runner/hardware_after_GPU.txt"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -u experiments/tianshu_native_int8_attention42_20261022/verify_cpu.py > "$projection_runner/CPU.log" 2>&1
