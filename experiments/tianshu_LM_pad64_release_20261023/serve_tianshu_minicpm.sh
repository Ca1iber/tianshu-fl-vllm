#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$base/../.." && pwd -P)
template="$base/serve_profile_template.json"
gate="$base/../tianshu_LM_pad64_official_results_20261023/OFFICIAL_GATE_32g.json"
dry=${TIANSHU_NATIVE_INT8_PROFILE_DRY_RUN:-0}
record_root=${TIANSHU_NATIVE_INT8_PROFILE_RECORD_DIR:-/workspace/benchmark-results/native_int8_LM_pad64_$(date -u +%Y%m%d_%H%M%S)_$$}
record_root=$(realpath -m -- "$record_root")
test "$#" -eq 0;test "$repo" = /workspace/vllm-plugin-FL
test -f "$base/QUALIFIED_RELEASE.sha256"
sha256sum --check --status "$base/QUALIFIED_RELEASE.sha256"
jq -e --slurpfile template "$template" '.complete and .promotion_passed and .official_benchmark and .zero_preemptions_verified and (.C64_original_experiment_passed==false) and .C64_CPU_runtime_audit_recovered and .all_inherited_runtime_and_new_Down_QKV_O_audits_recomputed and .C8_correct>=100 and .C64_correct>=100 and .actual_native_Down_runtime_all42_M4096 and .M64_Down_BF16_retained and .source_count==925 and .environment_count==63 and .model_frozen_sources==$template[0].frozen_sources and .server_command==$template[0].server_command and .controller_sources==$template[0].controller_sources and (.profile_environment+.additional_environment)==$template[0].environment' "$gate" >/dev/null
jq -e '(.environment|length)==63 and (.frozen_sources|length)==925 and (.controller_sources|length)>0 and (.server_command|length)==13 and .working_directory=="/workspace" and .environment.VLLM_PLUGINS=="fl" and .environment.PYTHONPATH=="/workspace/vllm-plugin-FL/experiments/tianshu_LM_pad64_service_20261022/hooks_candidate" and .environment.VLLM_DISABLE_COMPILE_CACHE=="1" and .environment.TIANSHU_FUTURE_KV_MODE=="candidate" and (.environment.TRITON_CACHE_DIR|startswith("/workspace/benchmark-results/"))' "$template" >/dev/null
launch_tmp=$(mktemp -d /tmp/tianshu-LM-pad64-launch.XXXXXX)
cleanup(){ rm -f "$launch_tmp/profile.json" "$launch_tmp/sources.nul" "$launch_tmp/environment.nul" "$launch_tmp/command.nul" "$launch_tmp/actual.json";rmdir "$launch_tmp"; }
trap cleanup EXIT
jq --arg root "$record_root" '.old_output_root as $old|.environment|with_entries(.value|=(split($old+"/")|join($root+"/")))' "$template" > "$launch_tmp/profile.json"
verified=0;controllers=0
for kind in frozen_sources controller_sources;do
    jq -j --arg kind "$kind" '.[$kind]|to_entries[]|.value,"\u0000",.key,"\u0000"' "$template" > "$launch_tmp/sources.nul"
    count=0
    while IFS= read -r -d '' expected && IFS= read -r -d '' filename;do
        test -f "$filename" || { printf 'Missing frozen file: %s\n' "$filename" >&2;exit 1; }
        actual=$(sha256sum < "$filename");actual=${actual%% *}
        [[ "$actual" == "$expected" ]] || { printf 'Source SHA mismatch: %s\n' "$filename" >&2;exit 1; }
        count=$((count+1))
    done < "$launch_tmp/sources.nul"
    if [[ "$kind" == frozen_sources ]];then verified=$count;else controllers=$count;fi
done
test "$verified" -eq 925;test "$controllers" -eq "$(jq '.controller_sources|length' "$template")"
while IFS= read -r name;do
    case "$name" in TIANSHU_*|VLLM_FL_*|FLAGGEMS_ILUVATAR_*|VLLM_PLUGINS|VLLM_DISABLE_COMPILE_CACHE|PYTHONPATH|PYTHONUNBUFFERED|TRITON_PRINT_AUTOTUNING|TRITON_CACHE_DIR|CUDA_VISIBLE_DEVICES)unset "$name";;esac
done < <(compgen -e)
jq -j 'to_entries[]|.key,"\u0000",.value,"\u0000"' "$launch_tmp/profile.json" > "$launch_tmp/environment.nul"
while IFS= read -r -d '' name && IFS= read -r -d '' value;do export "$name=$value";done < "$launch_tmp/environment.nul"
jq -j '.server_command[]|.,"\u0000"' "$template" > "$launch_tmp/command.nul"
mapfile -d '' -t command < "$launch_tmp/command.nul"
env -0|jq -Rs 'split("\u0000")|map(select(length>0)|capture("^(?<key>[^=]+)=(?<value>.*)$";"s"))|map(select((.key|startswith("TIANSHU_") or startswith("VLLM_FL_") or startswith("FLAGGEMS_ILUVATAR_")) or (.key|IN("VLLM_PLUGINS","VLLM_DISABLE_COMPILE_CACHE","PYTHONPATH","PYTHONUNBUFFERED","TRITON_PRINT_AUTOTUNING","TRITON_CACHE_DIR","CUDA_VISIBLE_DEVICES"))))|from_entries' > "$launch_tmp/actual.json"
jq -e --slurpfile actual "$launch_tmp/actual.json" '.==$actual[0]' "$launch_tmp/profile.json" >/dev/null
if [[ "$dry" == 1 ]];then
    jq -n --slurpfile template "$template" --slurpfile profile "$launch_tmp/profile.json" --slurpfile actual "$launch_tmp/actual.json" --arg root "$record_root" --argjson verified "$verified" --argjson controllers "$controllers" '{command:$template[0].server_command,working_directory:$template[0].working_directory,profile_environment:$profile[0],selected_actual_environment:$actual[0],record_root:$root,verified_runtime_source_count:$verified,verified_controller_source_count:$controllers,source_validation_passed:true,server_not_started:true}'
else
    mkdir -- "$record_root"
    jq -n --slurpfile template "$template" --slurpfile profile "$launch_tmp/profile.json" --argjson verified "$verified" --argjson controllers "$controllers" '{command:$template[0].server_command,working_directory:$template[0].working_directory,profile_environment:$profile[0],verified_runtime_source_count:$verified,verified_controller_source_count:$controllers}' > "$record_root/launcher_profile.json"
    cleanup;trap - EXIT
    cd /workspace
    exec "${command[@]}"
fi
