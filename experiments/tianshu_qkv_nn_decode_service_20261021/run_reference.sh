#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_qkv_nn_decode_service_20261021
old=experiments/tianshu_m_range_prefill_service_v2_20261021
runner=/workspace/benchmark-results/20261021_qkv_NN_direct_reference_runner_tianshu
root=/workspace/benchmark-results/20261021_qkv_NN_direct_reference_tianshu
test ! -e "$runner";test ! -e "$root"
ixsmi | grep -q 'No running processes found'
mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt";printf 'direct_471_reference\n' > "$runner/phase.txt"
trap 'code=$?;printf "%s\n" "$code" > "$runner/exit_code.txt";if (( code!=0 ));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";fi' EXIT
sha256sum "$base/PLAN_32g.md" "$base/run_reference.sh" "$old/run_service_trial.py" "$old/candidate_profile.py" > "$runner/frozen_reference_entrypoints.sha256"
timeout -k 30s 1800s env -u PYTHONPATH python3 -u "$old/run_service_trial.py" --mode candidate --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/service.log" 2>&1
sha256sum --check --quiet "$runner/frozen_reference_entrypoints.sha256"
jq -e '.phase=="complete" and (.screen_rounds|length)==3 and (.frozen_sources|length)==471' "$root/result.json" >/dev/null
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
