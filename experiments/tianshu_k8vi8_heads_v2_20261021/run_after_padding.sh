#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
waiter=/workspace/benchmark-results/20261021_dense_tile_padding_runner_tianshu
while :;do
 if (( $(date -u +%s)>=1792595880 ));then exit 0;fi
 phase=$(cat "$waiter/phase.txt" 2>/dev/null || true)
 case "$phase" in complete|failed)break;; esac
 sleep 15
done
exec bash experiments/tianshu_k8vi8_heads_v2_20261021/run_component.sh
