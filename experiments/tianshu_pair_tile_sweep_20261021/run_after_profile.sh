#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
previous=/workspace/benchmark-results/20261021_cpu_profile_qualified_runner_tianshu
queue=/workspace/benchmark-results/20261021_pair_tile_sweep_queue_tianshu
mkdir "$queue";printf 'waiting_for_profile_terminal\n' > "$queue/phase.txt"
while :;do phase=$(cat "$previous/phase.txt");case "$phase" in complete|failed)break;;esac;sleep 10;done
printf 'starting_component\n' > "$queue/phase.txt"
exec bash experiments/tianshu_pair_tile_sweep_20261021/run_component.sh
