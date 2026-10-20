#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_prefill_int8_validation_20261021
screen=experiments/tianshu_prefill_int8_service_20261021
runner=/workspace/benchmark-results/20261021_prefill_int8_validation_runner_tianshu
math_root=/workspace/benchmark-results/20261021_prefill_int8_math_tianshu
official_root=/workspace/benchmark-results/20261021_prefill_int8_official4k_tianshu
jq -e '.complete and .independently_rechecked and .gain_percent>0.5 and (.gain_percent>([.summaries[].spread_percent]|max)) and ([.summaries[].postfirst_autotuning_records]|all(.==0))' "$screen/SCREEN_GATE_32g.json" >/dev/null
test ! -e "$runner";test ! -e "$math_root";test ! -e "$official_root";mkdir "$runner";printf 'math\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed_or_quality_rejected\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
ixsmi > "$runner/hardware_before.txt";grep -q 'No running processes found' "$runner/hardware_before.txt"
timeout -k 30s 14400s env -u PYTHONPATH python3 -u "$base/run_validation.py" --stage math --output "$math_root" --screen-root "$screen" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/math.log" 2>&1
jq -e '.complete and .total==105 and .accuracy>0.95 and .actual_prefill_INT8_all_42_layers_verified' "$math_root/accuracy_audit.json" >/dev/null
printf 'official4k\n' > "$runner/phase.txt"
timeout -k 30s 7200s env -u PYTHONPATH python3 -u "$base/run_validation.py" --stage official --output "$official_root" --screen-root "$screen" --math-root "$math_root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/official4k.log" 2>&1
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
