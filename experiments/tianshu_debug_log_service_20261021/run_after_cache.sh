#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
previous=/workspace/benchmark-results/20261021_index_config_cache_service_pair_runner_tianshu
queue=/workspace/benchmark-results/20261021_debug_log_service_queue_tianshu
mkdir "$queue";printf 'waiting_for_cache_pair_terminal\n' > "$queue/phase.txt"
while :;do phase=$(cat "$previous/phase.txt");case "$phase" in complete|failed)break;;esac;sleep 10;done
printf 'starting_debug_log_pair\n' > "$queue/phase.txt"
exec bash experiments/tianshu_debug_log_service_20261021/run_pair.sh
