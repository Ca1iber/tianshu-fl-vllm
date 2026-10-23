#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$base/../.." && pwd -P)
test "$#" -le 1
mode=${1:-restore};case "$mode" in restore|--check);;*)exit 2;;esac
sha256sum --check --status "$base/QUALIFIED_RELEASE.sha256"
bash "$base/../tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh" "$@"
manifest="$base/../tianshu_LM_pad64_quality_20261022/LM_COMPONENT_CACHE.sha256"
cache=/workspace/benchmark-results/20261022_prefill_control_binary_all_GPU_tianshu/qualified_model_cache
missing=0
while read -r expected filename;do
 case "$filename" in "$cache"/CC465OJPQC6NIUG7OQ4S4K656JNKIXFO2UQZL3LBSDJTCMJGJ7HA/*|"$cache"/WH2KTRV74VEZKGIBK2W6AVJIRAGC7YPOI6KF42A2JNBMPMXPUOHQ/*);;*)exit 1;;esac
 if [ -f "$filename" ];then actual=$(sha256sum < "$filename");test "${actual%% *}" = "$expected";else missing=$((missing+1));fi
done < "$manifest"
if [ "$missing" -gt 0 ];then
 test "$mode" != --check
 temp=$(mktemp -d /tmp/tianshu-LM-cache.XXXXXX)
 trap 'rm -rf -- "$temp"' EXIT
 tar -xzf "$base/evidence/LM_component_cache.tar.gz" -C "$temp"
 while read -r expected filename;do
  [ -f "$filename" ] && continue
  relative=${filename#"$cache"/}
  actual=$(sha256sum < "$temp/$relative");test "${actual%% *}" = "$expected"
  mkdir -p -- "$(dirname -- "$filename")"
  cp -- "$temp/$relative" "$filename"
 done < "$manifest"
fi
sha256sum --check --status "$manifest"
printf 'New LM component cache all16 files verified, originally_missing=%s\n' "$missing"
