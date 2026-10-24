#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
ti_results=experiments/tianshu_W4_QKVO64_fitted_results_20261024
ti_base=/dev/shm/tianshu_QKVO64_model_profile_CPU_20261024
ti_temp=/dev/shm/20261024_QKVO64_immutable_SDK_snapshot.tar.xz
ti_members=/dev/shm/20261024_QKVO64_immutable_SDK_members.txt
ti_dest=/workspace/benchmark-results/20261024_QKVO64_immutable_SDK_snapshot.tar.xz
jq -e '.stage=="math"' /workspace/benchmark-results/20261024_W4_QKVO64_fitted_v2_math64_tianshu/state.json > /dev/null
test ! -e "$ti_temp"
jq -r '.references[].immutable_snapshot | sub("^/dev/shm/tianshu_QKVO64_model_profile_CPU_20261024/";"")' experiments/tianshu_W4_QKVO64_fitted_service_20261024/IMMUTABLE_SDK_REFERENCES_32g.json > "$ti_members"
XZ_OPT='-T2 --lzma2=preset=6,dict=64MiB' tar -C "$ti_base" -cJf "$ti_temp" -T "$ti_members"
tar -C "$ti_base" -dJf "$ti_temp"
ti_bytes=$(stat -c %s "$ti_temp")
ti_free=$(df --output=avail -B1 /workspace | tail -1)
if test "$ti_free" -gt "$((ti_bytes+268435456))"; then
 cp "$ti_temp" "$ti_dest"
 cmp "$ti_temp" "$ti_dest"
 sha256sum "$ti_dest" > "$ti_results/SDK_RAW_ARCHIVE_32g.txt"
 printf 'storage=persistent_workspace\n' >> "$ti_results/SDK_RAW_ARCHIVE_32g.txt"
else
 sha256sum "$ti_temp" > "$ti_results/SDK_RAW_ARCHIVE_32g.txt"
 printf 'storage=memory_pending_workspace_capacity\n' >> "$ti_results/SDK_RAW_ARCHIVE_32g.txt"
fi
printf 'bytes=%s\ntar_compare=passed\n' "$ti_bytes" >> "$ti_results/SDK_RAW_ARCHIVE_32g.txt"
cp "$ti_members" "$ti_results/SDK_ARCHIVE_MEMBERS_32g.txt"
cat "$ti_results/SDK_RAW_ARCHIVE_32g.txt"
