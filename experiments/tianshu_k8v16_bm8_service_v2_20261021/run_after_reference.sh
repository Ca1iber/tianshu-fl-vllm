#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_k8v16_bm8_service_v2_20261021
results=experiments/tianshu_k8v16_bm8_service_results_v2_20261021
runner=/workspace/benchmark-results/20261021_BM8_candidate_v2_runner_tianshu
root=/workspace/benchmark-results/20261021_BM8_candidate_v2_tianshu
test ! -e "$runner";test ! -e "$root"
jq -e '.complete and .CUDA_not_initialized and .original471_exact_subset' "$base/CPU_GATE_32g.json" >/dev/null
mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt";printf 'waiting_for_reference\n' > "$runner/phase.txt"
trap 'code=$?;printf "%s\n" "$code" > "$runner/exit_code.txt";if (( code!=0 ));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";fi' EXIT
sha256sum "$base"/*.py "$base"/*.sh "$base"/hooks_candidate/sitecustomize.py "$base"/archived_generated/*.py "$base/PLAN_32g.md" "$base/selected_plan.json" "$base/generated_sources.json" "$base/CPU_GATE_32g.json" "$results/verify_pair_cpu.py" > "$runner/frozen_entrypoints.sha256"
while true;do state=$(ps -p 306332 -o stat= || true);if test -z "$state" || [[ "$state" == Z* ]];then break;fi;sleep 10;done
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
jq -e '.phase=="complete" and (.frozen_sources|length)==471' /workspace/benchmark-results/20261021_BM8_direct_reference_tianshu/result.json >/dev/null
while true;do state=$(ps -p 307301 -o stat= || true);if test -z "$state" || [[ "$state" == Z* ]];then break;fi;sleep 5;done
ixsmi | grep -q 'No running processes found'
printf 'candidate_GPU\n' > "$runner/phase.txt"
timeout -k 30s 1200s env -u PYTHONPATH python3 -u "$base/run_service_trial.py" --mode candidate --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/service.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
jq -e '.phase=="complete" and (.screen_rounds|length)==3' "$root/result.json" >/dev/null
printf 'GPU_complete_pending_CPU_comparison\n' > "$runner/phase.txt"
env CUDA_VISIBLE_DEVICES='' VLLM_PLUGINS='' PYTHONPATH='' python3 -u "$results/verify_pair_cpu.py" > "$runner/CPU_comparison.log" 2>&1
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
