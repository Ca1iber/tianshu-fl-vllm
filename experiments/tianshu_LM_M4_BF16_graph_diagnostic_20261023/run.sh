#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_LM_M4_BF16_graph_diagnostic_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
ixsmi | grep 'No running processes found'
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 timeout --signal=TERM --kill-after=15s 300s python3 "$base/probe_gpu.py"
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 "$base/verify_cpu.py"
sha256sum --check --status "$base/EXECUTION.sha256"
ixsmi | grep 'No running processes found'
