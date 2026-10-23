#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_native_INT4_inventory_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ));test "$remaining" -gt 300
ixsmi | grep 'No running processes found'
trap 'ixsmi > /workspace/benchmark-results/20261023_native_INT4_smoke_hardware_after_tianshu.txt' EXIT
timeout --signal=TERM --kill-after=10s 90s python3 "$base/smoke_gpu.py"
CUDA_VISIBLE_DEVICES='' python3 "$base/verify_cpu.py"
