#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
waiter=/workspace/benchmark-results/20261020_lm_precision_validation_runner_v2_tianshu
while :;do
 if (( $(date -u +%s)>=1792597080 ));then exit 0;fi
 phase=$(cat "$waiter/phase.txt" 2>/dev/null || true)
 case "$phase" in complete|failed|screen_regression)break;; esac
 sleep 15
done
exec bash experiments/tianshu_dense_tile_padding_20261021/run_component.sh
