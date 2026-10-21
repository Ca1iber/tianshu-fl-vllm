#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_qkv_prefill_exact_reference_20261021
screen=experiments/tianshu_qkv_prefill_int8_service_v2_20261021
runner=/workspace/benchmark-results/20261021_qkv_prefill_exact407_runner_tianshu
root=/workspace/benchmark-results/20261021_qkv_prefill_exact407_reference_tianshu
jq -e '.complete and .predeclared_service_gate_passed' "$screen/SCREEN_GATE_32g.json" >/dev/null
jq -e '.complete and .all_407_sources_exact_subset_of_candidate' "$base/CPU_GATE_32g.json" >/dev/null
test ! -e "$runner";test ! -e "$root";mkdir "$runner"
printf 'reference407\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
ixsmi > "$runner/hardware_before.txt";grep -q 'No running processes found' "$runner/hardware_before.txt"
sha256sum "$base"/*.py "$base"/*.sh "$base/CPU_GATE_32g.json" "$base/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
timeout -k 30s 3600s env -u PYTHONPATH python3 -u experiments/tianshu_prefill_int8_service_20261021/run_service_trial.py --output "$root" --mode candidate --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/reference.log" 2>&1
printf 'independent_CPU_recheck\n' > "$runner/phase.txt"
env -u PYTHONPATH python3 -u "$base/verify_reference.py" > "$runner/verify.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
