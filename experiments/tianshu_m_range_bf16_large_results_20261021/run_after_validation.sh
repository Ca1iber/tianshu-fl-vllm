#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_m_range_bf16_large_results_20261021
probe=experiments/tianshu_m_range_bf16_large_20261021
parent=/workspace/benchmark-results/20261021_m_range_prefill_validation_v2_runner_tianshu
runner=/workspace/benchmark-results/20261021_m_range_BF16_large_runner_tianshu
root=/workspace/benchmark-results/20261021_m_range_BF16_large_GPU_tianshu
test ! -e "$runner";test ! -e "$root"
mkdir "$runner";printf '%s\n' "$$" > "$runner/runner_pid.txt";printf 'waiting_for_validation\n' > "$runner/phase.txt"
trap 'code=$?;printf "%s\n" "$code" > "$runner/exit_code.txt";if (( code!=0 ));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";fi' EXIT
sha256sum "$probe/probe_gpu.py" "$probe/CPU_GATE_32g.json" "$probe/PLAN_32g.md" "$base"/*.py "$base"/*.sh "$base/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
parent_pid=$(cat "$parent/runner_pid.txt")
while true;do
 state=$(ps -p "$parent_pid" -o stat= | tr -d ' ' || true)
 if [ -z "$state" ] || [[ "$state" == Z* ]];then break;fi
 if test "$(date -u +%s)" -ge "$(date -u -d '2026-10-21 15:15:00 UTC' +%s)";then printf 'not_started_insufficient_time\n' > "$runner/phase.txt";exit 0;fi
 sleep 15
done
if test "$(cat "$parent/phase.txt")" != complete;then printf 'not_started_parent_validation_failed\n' > "$runner/phase.txt";exit 0;fi
remaining=$(( $(date -u -d '2026-10-21 15:58:00 UTC' +%s)-$(date -u +%s) ))
if (( remaining<1800 ));then printf 'not_started_insufficient_time\n' > "$runner/phase.txt";exit 0;fi
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
ixsmi | grep -q 'No running processes found'
printf 'BF16_large_component\n' > "$runner/phase.txt"
limit=5400;if (( remaining<limit ));then limit=$remaining;fi
timeout -k 30s "${limit}s" env -u PYTHONPATH python3 -u "$probe/probe_gpu.py" --output "$root" > "$runner/component.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
ixsmi > "$runner/hardware_after_GPU.txt";grep -q 'No running processes found' "$runner/hardware_after_GPU.txt"
printf 'independent_CPU_verification\n' > "$runner/phase.txt"
env CUDA_VISIBLE_DEVICES='' VLLM_PLUGINS='' PYTHONPATH='' python3 -u "$base/verify_cpu.py" > "$runner/CPU_verification.log" 2>&1
sed -i 's/\r$//' "$base/ALL_TIMINGS_32g.csv"
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
