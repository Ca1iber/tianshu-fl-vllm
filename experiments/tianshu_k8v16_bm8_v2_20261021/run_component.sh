#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_k8v16_bm8_v2_20261021
runner=/workspace/benchmark-results/20261021_k8v16_BM8_v2_runner_tianshu
root=/workspace/benchmark-results/20261021_k8v16_BM8_v2_GPU_tianshu
reader=/workspace/benchmark-results/20261020_k8v16_pair_load_GPU_tianshu/pair_rows_reader/generated_pair_rows.py
test ! -e "$runner";test ! -e "$root"
jq -e '.complete and .coverage_cases==2816 and .actual_live_and_zero_query_mapping_cases==768' "$base/CPU_GATE_32g.json" >/dev/null
mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt";printf 'waiting_for_stage_diagnostic\n' > "$runner/phase.txt"
trap 'code=$?;printf "%s\n" "$code" > "$runner/exit_code.txt";if (( code!=0 ));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";fi' EXIT
sha256sum "$base"/*.py "$base"/*.sh "$base/CPU_GATE_32g.json" "$base/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
while true;do state=$(ps -p 304437 -o stat= || true);if test -z "$state" || [[ "$state" == Z* ]];then break;fi;sleep 10;done
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
ixsmi | grep -q 'No running processes found'
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s)-$(date -u +%s) ));test "$remaining" -gt 1200
printf 'component_GPU\n' > "$runner/phase.txt"
timeout -k 30s 1800s env -u PYTHONPATH python3 -u "$base/probe_gpu.py" --output "$root" --reader-source "$reader" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/GPU.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
printf 'independent_CPU_verification\n' > "$runner/phase.txt"
env CUDA_VISIBLE_DEVICES='' VLLM_PLUGINS='' PYTHONPATH='' python3 -u "$base/verify_gpu.py" "$root" > "$runner/CPU_verify.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
