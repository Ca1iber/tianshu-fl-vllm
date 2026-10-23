#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_numeric_archive_dedup_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
for pid in 26165 27948 30742 34801 34982;do if [ -e "/proc/$pid/stat" ] && [ "$(awk '{print $3}' "/proc/$pid/stat")" != Z ];then exit 1;fi;done
ixsmi | grep 'No running processes found'
runner=/workspace/benchmark-results/20261023_numeric_archive_dedup_runner_tianshu
mkdir "$runner"
jq -r '(.frozen_sources+.controller_sources)|to_entries[]|.value+"  "+.key' /workspace/benchmark-results/20261023_RMS_LM_short4k_candidate_tianshu/result.json > "$runner/PROTECTED_INPUTS.sha256"
jq -r '.files|to_entries[]|.value+"  "+.key' experiments/tianshu_prefill_control_binary_all_20261022/QUALIFIED_MODEL_CACHE_32g.json >> "$runner/PROTECTED_INPUTS.sha256"
cat experiments/tianshu_prefill_RMS_A8_QKV_20261023/GPU_INPUTS.sha256 >> "$runner/PROTECTED_INPUTS.sha256"
sha256sum --check --status "$runner/PROTECTED_INPUTS.sha256"
sha256sum --check --status experiments/tianshu_RMS_LM_quality_20261023/LM_COMPONENT_CACHE.sha256
CUDA_VISIBLE_DEVICES='' python3 "$base/deduplicate_cpu.py" > "$runner/dedup.log" 2>&1
sha256sum --check --status "$runner/PROTECTED_INPUTS.sha256"
sha256sum --check --status experiments/tianshu_RMS_LM_quality_20261023/LM_COMPONENT_CACHE.sha256
jq --argjson count "$(wc -l < "$runner/PROTECTED_INPUTS.sha256")" '.+{protected_source_controller_cache_inputs:$count,protected_SHA_before_after:true,LM16_cache_verified_before_after:true,all_previous_GPU_pipelines_terminal:true}' /workspace/benchmark-results/20261023_numeric_archive_dedup_CPU_tianshu/GATE_32g.json > "$base/GATE_32g.json"
