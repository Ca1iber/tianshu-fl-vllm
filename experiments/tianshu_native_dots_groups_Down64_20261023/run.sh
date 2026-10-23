#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_native_dots_groups_Down64_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ));test "$remaining" -gt 1200
ixsmi | grep 'No running processes found'
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 timeout --signal=TERM --kill-after=15s 900s python3 "$base/probe_gpu.py"
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 "$base/verify_cpu.py"
sha256sum --check --status "$base/EXECUTION.sha256"
ixsmi | grep 'No running processes found'
