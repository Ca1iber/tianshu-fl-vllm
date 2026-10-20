#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
queue=/workspace/benchmark-results/20261020_cold_start_v2_comparison_runner_tianshu
while :;do
 phase=$(cat "$queue/phase.txt" 2>/dev/null || true)
 case "$phase" in complete)break;; failed)exit 1;; esac
 sleep 15
done
exec bash experiments/tianshu_prefill_linear_service_20261020/run_prefill_pair.sh
