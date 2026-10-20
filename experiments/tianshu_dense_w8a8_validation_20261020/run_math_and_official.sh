#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_dense_w8a8_validation_20261020
root=/workspace/benchmark-results/20261020_dense_w8a8_validation_pipeline_tianshu
math=/workspace/benchmark-results/20261020_dense_w8a8_math_tianshu
official=/workspace/benchmark-results/20261020_dense_w8a8_official4k_tianshu
screen=/workspace/benchmark-results/20261020_dense_w8a8_service4k_v3_tianshu
remaining=$(( $(date -u -d '2026-10-20 15:58:00 UTC' +%s) - $(date -u +%s) ))
test "$remaining" -gt 6000
test ! -e "$root";test ! -e "$math";test ! -e "$official"
test -f "$screen/complete.txt"
jq -e '.complete and .independently_rechecked and .gain_percent>=1' experiments/tianshu_dense_w8a8_service_20261020/DIAGNOSTIC_4K_GATE_32g.json >/dev/null
ixsmi | grep -q 'No running processes found'
mkdir "$root"
printf '%s\n' "$$" > "$root/runner_pid.txt"
date -Is > "$root/started.txt"
trap 'code=$?; if ((code!=0));then cp "$root/phase.txt" "$root/failed_phase.txt";printf "failed\n" > "$root/phase.txt";printf "%s\n" "$code" > "$root/exit_code.txt";fi' EXIT
printf 'math\n' > "$root/phase.txt"
timeout -k 30s 7200s env -u PYTHONPATH python3 -u "$base/run_validation.py" --output "$math" --screen-root "$screen" --stop-at-utc 2026-10-20T15:58:00+00:00 --stage math > "$root/math.log" 2>&1
jq -e '.complete and .total==105 and .accuracy>0.95 and .actual_C8_dense_integer_path_verified' "$math/accuracy_audit.json" > "$root/quality_gate.txt"
printf 'official4k\n' > "$root/phase.txt"
timeout -k 30s 3600s env -u PYTHONPATH python3 -u "$base/run_validation.py" --output "$official" --screen-root "$screen" --math-root "$math" --stop-at-utc 2026-10-20T15:58:00+00:00 --stage official > "$root/official.log" 2>&1
jq -e '.complete and .requests_successful==1024' "$official/comparison.json" > "$root/official_gate.txt"
printf 'complete\n' > "$root/phase.txt"
date -Is > "$root/complete.txt"
