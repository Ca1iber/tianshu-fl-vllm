#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_nn_silu_fx_screen_20261021
runner=/workspace/benchmark-results/20261021_nn_silu_fx_screen_runner_tianshu
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s) - $(date -u +%s) ))
test "$remaining" -gt 5400
test ! -e "$runner"
jq -e '.complete and .CUDA_not_initialized' "$base/CPU_GATE_32g.json" >/dev/null
ixsmi | grep -q 'No running processes found'
mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt"
trap 'exit_code=$?; if (( exit_code!=0 )); then cp "$runner/phase.txt" "$runner/failed_phase.txt"; printf "failed\n" > "$runner/phase.txt"; printf "%s\n" "$exit_code" > "$runner/exit_code.txt"; fi' EXIT
find "$base" -type f \( -name '*.py' -o -name '*.sh' -o -name 'CPU_GATE_32g.json' -o -name 'PLAN_32g.md' \) -print0 | sort -z | xargs -0 sha256sum > "$runner/frozen_entrypoints.sha256"
for arm in control candidate reference;do
 printf '%s\n' "$arm" > "$runner/phase.txt"
 root=/workspace/benchmark-results/20261021_nn_silu_fx_screen_${arm}_tianshu
 test ! -e "$root"
 entry=$base/run_service_trial.py
 mode=$arm
 if [ "$arm" = reference ];then entry=experiments/tianshu_prefill_int8_service_20261021/run_service_trial.py;mode=candidate;fi
 timeout -k 30s 1800s env -u PYTHONPATH python3 -u "$entry" --mode "$mode" --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/$arm.log" 2>&1
 sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
done
printf '%s\n' independent_CPU_verification > "$runner/phase.txt"
env CUDA_VISIBLE_DEVICES='' VLLM_PLUGINS='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 "$base/verify_screen.py" > "$runner/verification.log" 2>&1
ixsmi > "$runner/hardware_after.txt";grep -q 'No running processes found' "$runner/hardware_after.txt"
printf '%s\n' complete > "$runner/phase.txt";date -Is > "$runner/complete.txt"
