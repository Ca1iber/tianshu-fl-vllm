#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_dense_budget_audit_recovery_20261021
runner=/workspace/benchmark-results/20261021_dense_budget_v2_official_recovery_runner_tianshu
test ! -e "$runner";mkdir "$runner";printf 'checking_idle\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
ixsmi > "$runner/hardware_before.txt";grep -q 'No running processes found' "$runner/hardware_before.txt"
printf 'official4k\n' > "$runner/phase.txt"
timeout -k 30s 2400s env -u PYTHONPATH python3 -u "$base/recover.py" --output /workspace/benchmark-results/20261021_dense_budget_v2_official4k_tianshu --screen-root experiments/tianshu_dense_budget8192_v2_20261021 --math-root /workspace/benchmark-results/20261021_dense_budget_v2_math_tianshu --stage official --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/official4k.log" 2>&1
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
