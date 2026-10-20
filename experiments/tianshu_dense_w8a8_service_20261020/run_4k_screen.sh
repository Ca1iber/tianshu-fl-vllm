#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
grep -q '^class ModelRunnerFL(' /usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py
base=experiments/tianshu_dense_w8a8_service_20261020
root=/workspace/benchmark-results/20261020_dense_w8a8_service4k_v2_tianshu
remaining=$(( $(date -u -d '2026-10-20 15:58:00 UTC' +%s) - $(date -u +%s) ))
test "$remaining" -gt 2400
test ! -e "$root"
jq -e '.complete and .CPU_only' "$base/PROFILE_CPU_GATE_32g.json" >/dev/null
ixsmi | grep -q 'No running processes found'
mkdir "$root"
printf '%s\n' "$$" > "$root/runner_pid.txt"
date -Is > "$root/started.txt"
trap 'code=$?; if ((code!=0)); then cp "$root/phase.txt" "$root/failed_phase.txt"; printf "failed\n" > "$root/phase.txt"; printf "%s\n" "$code" > "$root/exit_code.txt"; fi' EXIT
sha256sum "$base/run_4k_screen.sh" "$base/verify_screen.py" "$base/PROFILE_CPU_GATE_32g.json" > "$root/controller_sources.sha256"
for mode in candidate control; do
    printf '%s\n' "$mode" > "$root/phase.txt"
    env -u PYTHONPATH timeout -k 30s 3600s python3 -u "$base/run_service_trial.py" --output "$root/$mode" --mode "$mode" --input-len 4096 --stop-at-utc 2026-10-20T15:58:00+00:00 > "$root/$mode.log" 2>&1
    jq -e '.phase=="complete"' "$root/$mode/result.json" >/dev/null
done
printf '%s\n' independent_screen_audit > "$root/phase.txt"
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 "$base/verify_screen.py" "$root" > "$root/audit.log" 2>&1
sha256sum -c "$root/controller_sources.sha256" > "$root/controller_verified.txt"
ixsmi > "$root/hardware_after.txt"
grep -q 'No running processes found' "$root/hardware_after.txt"
printf 'complete\n' > "$root/phase.txt"
date -Is > "$root/complete.txt"
