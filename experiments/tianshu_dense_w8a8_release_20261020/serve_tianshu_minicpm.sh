#!/usr/bin/env bash
set -euo pipefail
base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$base/../.." && pwd -P)"
template="$base/serve_profile_template.json"
dry="${TIANSHU_DENSE_W8A8_PROFILE_DRY_RUN:-0}"
record_root="${TIANSHU_DENSE_W8A8_PROFILE_RECORD_DIR:-/workspace/benchmark-results/dense_w8a8_profile_$(date -u +%Y%m%d_%H%M%S)_$$}"
record_root=$(realpath -m -- "$record_root")
test "$#" -eq 0
jq -e '(.environment|length)==57 and (.frozen_sources|length)==311 and (.controller_sources|length)==9 and (.server_command|length)==13 and .environment.VLLM_PLUGINS=="fl" and .environment.VLLM_DISABLE_COMPILE_CACHE=="1" and .environment.TIANSHU_DENSE_INT8_MODE=="candidate" and .environment.TIANSHU_PAIR_LOAD_MODE=="candidate" and .environment.TIANSHU_FUTURE_KV_MODE=="candidate" and (.environment|has("TIANSHU_DECODE_METADATA_MODE")|not)' "$template" >/dev/null
launch_tmp=$(mktemp -d /tmp/tianshu-dense-w8a8-launch.XXXXXX)
trap 'rm -f "$launch_tmp/profile.json" "$launch_tmp/sources.nul" "$launch_tmp/environment.nul" "$launch_tmp/command.nul" "$launch_tmp/actual.json";rmdir "$launch_tmp"' EXIT
jq --arg root "$record_root" --arg repo "$repo" '.old_output_root as $old|.old_repository_root as $old_repo|.environment|with_entries(.value|=(split($old+"/")|join($root+"/")|split($old_repo+"/")|join($repo+"/")))' "$template" > "$launch_tmp/profile.json"
verified=0;controllers=0
for kind in frozen_sources controller_sources; do
    jq -j --arg kind "$kind" --arg repo "$repo" '.old_repository_root as $old|.[$kind]|to_entries[]|.value,"\u0000",(.key|split($old+"/")|join($repo+"/")),"\u0000"' "$template" > "$launch_tmp/sources.nul"
    count=0
    while IFS= read -r -d '' expected && IFS= read -r -d '' filename; do
        if ! test -f "$filename";then printf 'Missing source: %s; check restore_generated_sources.sh.\n' "$filename" >&2;exit 1;fi
        actual=$(sha256sum < "$filename");actual=${actual%% *}
        if [[ "$actual" != "$expected" ]];then printf 'Source SHA mismatch: %s\n' "$filename" >&2;exit 1;fi
        count=$((count+1))
    done < "$launch_tmp/sources.nul"
    if [[ "$kind" == frozen_sources ]];then verified=$count;else controllers=$count;fi
done
test "$verified" -eq 311;test "$controllers" -eq 9
while IFS= read -r name;do
    case "$name" in TIANSHU_*|VLLM_FL_*|FLAGGEMS_ILUVATAR_*|VLLM_PLUGINS|VLLM_DISABLE_COMPILE_CACHE|PYTHONPATH|PYTHONUNBUFFERED|TRITON_PRINT_AUTOTUNING|CUDA_VISIBLE_DEVICES)unset "$name";;esac
done < <(compgen -e)
jq -j 'to_entries[]|.key,"\u0000",.value,"\u0000"' "$launch_tmp/profile.json" > "$launch_tmp/environment.nul"
while IFS= read -r -d '' name && IFS= read -r -d '' value;do export "$name=$value";done < "$launch_tmp/environment.nul"
jq -j '.server_command[]|.,"\u0000"' "$template" > "$launch_tmp/command.nul"
mapfile -d '' -t command < "$launch_tmp/command.nul"
env -0|jq -Rs 'split("\u0000")|map(select(length>0)|capture("^(?<key>[^=]+)=(?<value>.*)$";"s"))|map(select((.key|startswith("TIANSHU_") or startswith("VLLM_FL_") or startswith("FLAGGEMS_ILUVATAR_")) or (.key|IN("VLLM_PLUGINS","VLLM_DISABLE_COMPILE_CACHE","PYTHONPATH","PYTHONUNBUFFERED","TRITON_PRINT_AUTOTUNING","CUDA_VISIBLE_DEVICES"))))|from_entries' > "$launch_tmp/actual.json"
jq -e --slurpfile actual "$launch_tmp/actual.json" '.==$actual[0]' "$launch_tmp/profile.json" >/dev/null
if [[ "$dry" == 1 ]];then
    jq -n --slurpfile template "$template" --slurpfile profile "$launch_tmp/profile.json" --slurpfile actual "$launch_tmp/actual.json" --arg root "$record_root" --argjson verified "$verified" --argjson controllers "$controllers" '{command:$template[0].server_command,profile_environment:$profile[0],selected_actual_environment:$actual[0],record_root:$root,verified_runtime_source_count:$verified,verified_controller_source_count:$controllers,source_validation_passed:true,server_not_started:true}'
else
    mkdir -- "$record_root"
    jq -n --slurpfile template "$template" --slurpfile profile "$launch_tmp/profile.json" --argjson verified "$verified" --argjson controllers "$controllers" '{command:$template[0].server_command,profile_environment:$profile[0],verified_runtime_source_count:$verified,verified_controller_source_count:$controllers}' > "$record_root/launcher_profile.json"
    rm -f "$launch_tmp/profile.json" "$launch_tmp/sources.nul" "$launch_tmp/environment.nul" "$launch_tmp/command.nul" "$launch_tmp/actual.json";rmdir "$launch_tmp";trap - EXIT
    exec "${command[@]}"
fi
