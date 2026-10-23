#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_native_W4_gateup_component_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$base/probe_gpu.py"
ixsmi > /dev/shm/tianshu_W4_gateup_component_v2_20261023/GPU/hardware_after_exit.txt
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$base/verify_cpu.py"
