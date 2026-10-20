#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_k8v16_pair_validation_20261020
root=/workspace/benchmark-results/20261020_pair_K_validation_pipeline_tianshu
math=/workspace/benchmark-results/20261020_pair_K_math_tianshu
official=/workspace/benchmark-results/20261020_pair_K_official4k_tianshu
screen=/workspace/benchmark-results/20261020_k8v16_pair_service4k_v2_tianshu
test ! -e "$root"
test ! -e "$math"
test ! -e "$official"
test -f "$screen/complete.txt"
test -f /workspace/benchmark-results/20261020_k8v16_word_decode_runner_tianshu/complete.txt
ixsmi | grep -q 'No running processes found'
mkdir "$root"
printf '%s\n' "$$" > "$root/runner_pid.txt"
date -Is > "$root/started.txt"
trap 'validation_exit=$?; if (( validation_exit!=0 )); then cp "$root/phase.txt" "$root/failed_phase.txt"; printf "failed\n" > "$root/phase.txt"; printf "%s\n" "$validation_exit" > "$root/exit_code.txt"; fi' EXIT
printf 'math\n' > "$root/phase.txt"
timeout -k 30s 14400s env -u PYTHONPATH python3 -u "$base/run_validation.py" --output "$math" --screen-root "$screen" --stop-at-utc 2026-10-20T15:58:00+00:00 --stage math > "$root/math.log" 2>&1
jq -e '.complete and .total==105 and .accuracy>0.95' "$math/accuracy_audit.json" > "$root/quality_gate.txt"
printf 'official4k\n' > "$root/phase.txt"
timeout -k 30s 7200s env -u PYTHONPATH python3 -u "$base/run_validation.py" --output "$official" --screen-root "$screen" --math-root "$math" --stop-at-utc 2026-10-20T15:58:00+00:00 --stage official > "$root/official.log" 2>&1
jq -e '.complete and .requests_successful==1024' "$official/comparison.json" > "$root/official_gate.txt"
printf 'complete\n' > "$root/phase.txt"
date -Is > "$root/ended.txt"
printf 'complete\n' > "$root/complete.txt"
