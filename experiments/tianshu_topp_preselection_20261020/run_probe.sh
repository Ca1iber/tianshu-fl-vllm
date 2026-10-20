#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_topp_preselection_20261020
root=/workspace/benchmark-results/20261020_topp_preselection_GPU_tianshu
mkdir "$root"
printf '%s\n' "$$" > "$root/runner_pid.txt"
date -Iseconds > "$root/started.txt"
ixsmi > "$root/hardware_before.txt"
grep -q 'No running processes found' "$root/hardware_before.txt"
sha256sum "$base"/*.py "$base"/CPU_GATE_32g.json "$base"/run_probe.sh > "$root/frozen_entrypoints.sha256"
jq -r '.sources|to_entries[]|"\(.value)  \(.key)"' "$base/CPU_GATE_32g.json" > "$root/frozen_CPU_sources.sha256"
sha256sum --check --quiet "$root/frozen_CPU_sources.sha256"
if timeout -k 20s 2100s env -u PYTHONPATH VLLM_PLUGINS=fl python3 -u "$base/probe.py" --output "$root/micro" --stop-at-utc 2026-10-20T15:58:00+00:00 > "$root/probe.log" 2>&1; then
    probe_code=0
else
    probe_code=$?
fi
printf '%s\n' "$probe_code" > "$root/exit_code.txt"
ixsmi > "$root/hardware_after.txt"
date -Iseconds > "$root/ended.txt"
sha256sum --check --quiet "$root/frozen_entrypoints.sha256"
sha256sum --check --quiet "$root/frozen_CPU_sources.sha256"
grep -q 'No running processes found' "$root/hardware_after.txt"
test "$probe_code" -eq 0
jq -e '.complete and .numerical_checks_passed and (.relative_passing_plans|index("geo")!=null)' "$root/micro/result.json" > "$root/component_gate.txt"
date -Iseconds > "$root/complete.txt"
