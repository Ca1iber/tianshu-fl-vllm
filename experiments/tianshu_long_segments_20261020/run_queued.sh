#!/usr/bin/env bash
set -euo pipefail
repo=/workspace/vllm-plugin-FL
formal=/workspace/benchmark-results/20261020_k8v16_official16k_tianshu
formal_launcher=/workspace/benchmark-results/20261020_k8v16_official16k_launcher_tianshu
queue=/workspace/benchmark-results/20261020_after16k_queue_tianshu
mkdir "$queue"
cd "$repo"
sha256sum experiments/tianshu_long_segments_20261020/{source.py,check_cpu.py,component_launch.py,probe.py,CPU_GATE_32g.json} experiments/tianshu_weight_soft8_20261020/{codec.py,kernel.py,check_cpu.py,probe.py,CPU_GATE_32g.json} > "$queue/frozen_sources.sha256"
printf '%s\n' "$$" > "$queue/queue_pid.txt"
while test ! -f "$formal_launcher/ended.txt"; do sleep 15; done
test -f "$formal/benchmark_ended.txt"
csv=$(awk '/^Raw CSV: /{print $3}' "$formal/benchmark.log")
case "$csv" in benchmark_results/raw_runs_*.csv) ;; *) exit 1;; esac
awk -F, 'NR>1 {n++; if($1==16384 && $2==1024 && $3==64 && $4==128 && $5==128 && $6=="SUCCESS") ok++} END {exit !(n==4 && ok==4)}' "/workspace/$csv"
sha256sum --check "$queue/frozen_sources.sha256"
date --iso-8601=seconds > "$queue/started_components.txt"
env VLLM_PLUGINS=fl timeout -k 20s 1800s python3 -u experiments/tianshu_long_segments_20261020/probe.py \
    --output /workspace/benchmark-results/20261020_long_segments_tianshu \
    --reader-source /workspace/benchmark-results/20261019_k8v16_budget_default_official4k_tianshu/decode_coherent_records/cache_generated/97987/generated_cache_policy.py \
    --stop-at-utc 2026-10-20T15:58:00+00:00 > "$queue/long_segments.log" 2>&1
env VLLM_PLUGINS=fl timeout -k 20s 1800s python3 -u experiments/tianshu_weight_soft8_20261020/probe.py \
    --output /workspace/benchmark-results/20261020_weight_soft8_component_tianshu \
    --stop-at-utc 2026-10-20T15:58:00+00:00 > "$queue/weight_soft8.log" 2>&1
date --iso-8601=seconds > "$queue/complete.txt"
