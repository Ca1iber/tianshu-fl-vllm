#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
log=/workspace/benchmark-results/20261023_storage_between_LM_arms_tianshu
backup=/dev/shm/tianshu-storage-20261023
jq -e '.phase=="complete" and (.screen_rounds|length)==3' /workspace/benchmark-results/20261023_LM_pad64_short4k_control_tianshu/result.json >/dev/null
test "$(awk '{print $3}' /proc/1807/stat)" = T
ixsmi | grep 'No running processes found'
test ! -e "$log"
test ! -e "$backup"
mkdir "$log" "$backup"
date -Is > "$log/started.txt"
df -B1 /workspace > "$log/disk_before.txt"
jq -r '.model_frozen_sources|to_entries[]|.value+"  "+.key' experiments/tianshu_LM_pad64_quality_results_v2_20261022/QUALITY_GATE_32g.json > "$log/PROTECTED_MODEL_925.sha256"
! grep -q '/fxgraph/' "$log/PROTECTED_MODEL_925.sha256"
sha256sum --check --status "$log/PROTECTED_MODEL_925.sha256"
sha256sum --check --status experiments/tianshu_LM_pad64_quality_20261022/LM_COMPONENT_CACHE.sha256
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check > "$log/strict_cache_before.txt"
find /root/.cache/vllm -type f -path '*/fxgraph/*' -print0 > "$log/fxgraph_files.nul"
while IFS= read -r -d '' file;do
 case "$file" in /root/.cache/vllm/*/fxgraph/*);;*)exit 1;;esac
 case "$file" in *.py|*.cubin|*.so|*.ttir|*.ttgir|*.asm|*.json|*.source)exit 1;;esac
done < "$log/fxgraph_files.nul"
xargs -0 -r sha256sum -- < "$log/fxgraph_files.nul" > "$log/FXGRAPH_ARCHIVED_FILES.sha256"
tar --create --absolute-names --file="$backup/fxgraph_backup.tar" --null --files-from="$log/fxgraph_files.nul"
tar --compare --absolute-names --file="$backup/fxgraph_backup.tar"
sha256sum "$backup/fxgraph_backup.tar" > "$log/ARCHIVE.sha256"
xargs -0 -r rm -- < "$log/fxgraph_files.nul"
sha256sum --check --status "$log/PROTECTED_MODEL_925.sha256"
sha256sum --check --status experiments/tianshu_LM_pad64_quality_20261022/LM_COMPONENT_CACHE.sha256
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check > "$log/strict_cache_after.txt"
df -B1 /workspace > "$log/disk_after.txt"
wc -l "$log/FXGRAPH_ARCHIVED_FILES.sha256"
cat "$log/disk_before.txt" "$log/disk_after.txt" "$log/strict_cache_after.txt"
date -Is > "$log/complete.txt"
kill -CONT 1807
printf '%s\n' 'BACKUP_EXACT_TAR_COMPARE_PROTECTED_SOURCES_AND_CACHE_PASSED_PAIR_CONTINUED'
