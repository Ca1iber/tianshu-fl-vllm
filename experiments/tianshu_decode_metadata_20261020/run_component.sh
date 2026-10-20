#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
root=/workspace/benchmark-results/20261020_decode_metadata_GPU_tianshu
test ! -e "$root"
test -f /workspace/benchmark-results/20261020_future_kv_service16k_v2_tianshu/complete.txt
ixsmi | grep -q 'No running processes found'
mkdir "$root"
printf '%s\n' "$$" > "$root/runner_pid.txt"
date -Is > "$root/started.txt"
sha256sum experiments/tianshu_decode_metadata_20261020/*.py experiments/tianshu_decode_metadata_20261020/CPU_GATE_32g.json experiments/tianshu_decode_metadata_20261020/run_component.sh > "$root/sources.sha256"
set +e
export VLLM_PLUGINS=fl
timeout -k 30s 2100s python3 -u experiments/tianshu_decode_metadata_20261020/probe_gpu.py --output "$root/micro" --stop-at-utc 2026-10-20T15:58:00+00:00 > "$root/probe.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$root/exit_code.txt"
ixsmi > "$root/hardware_after.txt"
date -Is > "$root/ended.txt"
test "$code" -eq 0
sha256sum -c "$root/sources.sha256" > "$root/sources_verified.txt"
jq -e '.complete and (.cases|length)==14' "$root/micro/result.json" > "$root/component_gate.txt"
grep -q 'No running processes found' "$root/hardware_after.txt"
printf 'complete\n' > "$root/complete.txt"
