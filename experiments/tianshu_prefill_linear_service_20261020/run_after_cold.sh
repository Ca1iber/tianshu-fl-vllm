#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
queue=/workspace/benchmark-results/20261020_cold_start_comparison_runner_tianshu
while kill -0 84947 2>/dev/null;do sleep 15;done
test "$(cat "$queue/phase.txt")" = complete
exec bash experiments/tianshu_prefill_linear_service_20261020/run_prefill_pair.sh
