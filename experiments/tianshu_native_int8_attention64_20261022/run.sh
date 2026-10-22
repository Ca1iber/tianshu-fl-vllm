#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
python3 -u experiments/tianshu_native_int8_attention64_20261022/probe_gpu.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 experiments/tianshu_native_int8_attention64_20261022/verify_cpu.py
