#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_RMS_A8_GU_v2_C64_results_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
for pid in 18883 20683;do
 while [ -e "/proc/$pid/stat" ] && [ "$(awk '{print $3}' "/proc/$pid/stat")" != Z ];do
  test "$(date -u +%s)" -lt "$(date -u -d '2026-10-23 15:58:00 UTC' +%s)"
  sleep 10
 done
done
jq -e '.phase=="complete" and .evalscope_exit_code==0' /workspace/benchmark-results/20261023_RMS_A8_GU_v2_math64_tianshu/result.json >/dev/null
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 "$base/verify_cpu.py" --batches 64
