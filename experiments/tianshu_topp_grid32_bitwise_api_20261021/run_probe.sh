#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_topp_grid32_bitwise_api_20261021
root=/workspace/benchmark-results/20261021_topp_grid32_bitwise_api_GPU_tianshu
mkdir "$root"
printf '%s\n' "$$" > "$root/runner_pid.txt"
ixsmi > "$root/hardware_before.txt"
grep -q 'No running processes found' "$root/hardware_before.txt"
sha256sum "$base"/*.py "$base"/*.sh "$base"/hooks_*/*.py "$base"/CPU_GATE_32g.json "$base"/PLAN_32g.md > "$root/frozen_entrypoints.sha256"
if timeout -k 20s 2100s env -u PYTHONPATH python3 -u "$base/probe_api.py" --output "$root/micro" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$root/probe.log" 2>&1;then code=0;else code=$?;fi
printf '%s\n' "$code" > "$root/exit_code.txt"
ixsmi > "$root/hardware_after.txt"
sha256sum --check --quiet "$root/frozen_entrypoints.sha256"
grep -q 'No running processes found' "$root/hardware_after.txt"
test "$code" -eq 0
jq -e '.complete and .phase=="complete"' "$root/micro/result.json" > "$root/component_gate.txt"
date -Is > "$root/complete.txt"
