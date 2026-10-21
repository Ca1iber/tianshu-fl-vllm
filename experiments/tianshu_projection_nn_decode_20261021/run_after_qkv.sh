#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_projection_nn_decode_20261021
parent=/workspace/benchmark-results/20261021_QKV_NN_decode_runner_tianshu
runner=/workspace/benchmark-results/20261021_projection_NN_runner_tianshu
test ! -e "$runner";mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt";printf 'waiting_for_QKV\n' > "$runner/phase.txt"
trap 'code=$?;printf "%s\n" "$code" > "$runner/exit_code.txt";if (( code!=0 ));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";fi' EXIT
sha256sum "$base"/*.py "$base"/*.sh "$base/tiles.json" "$base/CPU_GATE_32g.json" "$base/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
pid=$(cat "$parent/runner_pid.txt")
while true;do
 state=$(ps -p "$pid" -o stat= | tr -d ' ' || true)
 if [ -z "$state" ] || [[ "$state" == Z* ]];then break;fi
 if (( $(date -u +%s)>= $(date -u -d '2026-10-21 15:33 UTC' +%s) ));then printf 'not_started_insufficient_time\n' > "$runner/phase.txt";exit 0;fi
 sleep 15
done
if [ "$(cat "$parent/phase.txt")" != complete ];then printf 'not_started_parent_not_complete\n' > "$runner/phase.txt";exit 0;fi
for role in o down;do
 root=/workspace/benchmark-results/20261021_projection_NN_${role}_GPU_tianshu
 test ! -e "$root"
 remaining=$(( $(date -u -d '2026-10-21 15:58 UTC' +%s)-$(date -u +%s) ))
 if (( remaining<1500 ));then printf 'not_started_%s_insufficient_time\n' "$role" > "$runner/phase.txt";exit 0;fi
 sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
 ixsmi | grep -q 'No running processes found'
 printf '%s_NN_component\n' "$role" > "$runner/phase.txt"
 timeout -k 30s 1800s env -u PYTHONPATH python3 -u "$base/probe_gpu.py" --projection "$role" --output "$root" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/component_${role}.log" 2>&1
 sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
 ixsmi > "$runner/hardware_after_${role}.txt";grep -q 'No running processes found' "$runner/hardware_after_${role}.txt"
 printf '%s_CPU_full_INT32\n' "$role" > "$runner/phase.txt"
 env CUDA_VISIBLE_DEVICES='' VLLM_PLUGINS='' PYTHONPATH='' python3 -u "$base/verify_cpu.py" --projection "$role" > "$runner/CPU_${role}.log" 2>&1
 sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
done
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
