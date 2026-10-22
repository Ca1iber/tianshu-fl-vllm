#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
recovery=experiments/tianshu_LM_pad64_audit_recovery_20261022
sha256sum --check --status "$recovery/EXECUTION.sha256"
while [ ! -f /workspace/benchmark-results/20261022_LM_pad64_math64_tianshu/result.json ]; do
 test "$(date -u +%s)" -lt "$(date -u -d '2026-10-22 15:58:00 UTC' +%s)"
 sleep 10
done
while [ -e /proc/116138/stat ] && [ "$(awk '{print $3}' /proc/116138/stat)" != Z ]; do sleep 1; done
jq -e '.phase=="audit_failed" and .evalscope_exit_code==0' /workspace/benchmark-results/20261022_LM_pad64_math64_tianshu/result.json >/dev/null
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 "$recovery/recover_C64_cpu.py" > /workspace/benchmark-results/20261022_LM_pad64_C64_CPU_recovery_tianshu.log 2>&1
jq -e '.complete and .accuracy_passed and .correct/105>0.95 and .new_actual_Math_guard_execution_traced and .all_other_runtime_checks_recomputed' "$recovery/RECOVERY_GATE_32g.json" >/dev/null
printf '%s\n' 'C64_ORIGINAL_FAILURE_PRESERVED_CPU_AUDIT_RECOVERED_ACCURACY_PASSED_STARTING_FIRST_C8'
if pgrep -f 'run_validation.py|run_service_trial.py|probe_gpu.py|/usr/local/bin/vllm serve|VLLM::EngineCore' >/dev/null; then exit 1; fi
ixsmi > /workspace/benchmark-results/20261022_LM_pad64_C8_hardware_before_launch_tianshu.txt
grep 'No running processes found' /workspace/benchmark-results/20261022_LM_pad64_C8_hardware_before_launch_tianshu.txt
sha256sum --check --status experiments/tianshu_LM_pad64_quality_20261022/EXECUTION.sha256
python3 experiments/tianshu_LM_pad64_quality_20261022/run_validation.py --output /workspace/benchmark-results/20261022_LM_pad64_math8_tianshu --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_LM_pad64_20261022 --stop-at-utc 2026-10-22T15:58:00+00:00 --stage math --batch 8 --down-scope prefill > /workspace/benchmark-results/20261022_LM_pad64_quality_runner_tianshu/math8.log 2>&1
