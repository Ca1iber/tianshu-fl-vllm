#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
waiter=/workspace/benchmark-results/20261020_dense_dynamic_batches_runner_tianshu
while :;do
 if (( $(date -u +%s)>=1792597080 ));then exit 0;fi
 phase=$(cat "$waiter/phase.txt" 2>/dev/null || true)
 case "$phase" in complete|failed)break;; esac
 sleep 15
done
exec bash experiments/tianshu_dense_runtime_m_20261020/run_component.sh
