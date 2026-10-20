#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
queue=/workspace/benchmark-results/20261020_dynamic_k8_codecs_queue_tianshu
mkdir "$queue"
sha256sum experiments/tianshu_dynamic_k8_intdot_20261020/{codec.py,writer.py,probe_codecs_gpu.py,CPU_GATE_32g.json} > "$queue/frozen_sources.sha256"
printf '%s\n' "$$" > "$queue/queue_pid.txt"
while test ! -f /workspace/benchmark-results/20261020_long_segments_service16k_tianshu/comparison.json; do sleep 5; done
sha256sum --check "$queue/frozen_sources.sha256"
date --iso-8601=seconds > "$queue/started.txt"
env VLLM_PLUGINS=fl timeout -k 20s 600s python3 -u experiments/tianshu_dynamic_k8_intdot_20261020/probe_codecs_gpu.py \
 --output /workspace/benchmark-results/20261020_dynamic_k8_gpu_codecs_tianshu \
 --stop-at-utc 2026-10-20T15:58:00+00:00 > "$queue/codecs.log" 2>&1
date --iso-8601=seconds > "$queue/complete.txt"
