#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_lm_precision_validation_20261020
pair=/workspace/benchmark-results/20261020_lm_precision_pair_runner_tianshu
runner=/workspace/benchmark-results/20261020_lm_precision_validation_runner_tianshu
test ! -e "$runner";mkdir "$runner";printf 'waiting_for_screen\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
while :;do
 phase=$(cat "$pair/phase.txt" 2>/dev/null || true)
 case "$phase" in complete)break;; failed)exit 1;; esac
 sleep 15
done
printf 'verifying_screen\n' > "$runner/phase.txt"
env -u PYTHONPATH python3 "$base/verify_screen.py" --runner "$pair" > "$runner/screen_gate.log" 2>&1
if test "$(jq -r .quality_worth_running "$pair/SCREEN_GATE_32g.json")" != true;then printf 'screen_regression\n' > "$runner/phase.txt";exit 0;fi
printf 'math\n' > "$runner/phase.txt"
timeout -k 30s 7200s env -u PYTHONPATH python3 -u "$base/run_validation.py" --output /workspace/benchmark-results/20261020_lm_precision_math_tianshu --screen-root "$pair" --stage math --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/math.log" 2>&1
printf 'official4k\n' > "$runner/phase.txt"
timeout -k 30s 2400s env -u PYTHONPATH python3 -u "$base/run_validation.py" --output /workspace/benchmark-results/20261020_lm_precision_official4k_tianshu --screen-root "$pair" --math-root /workspace/benchmark-results/20261020_lm_precision_math_tianshu --stage official --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/official4k.log" 2>&1
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
