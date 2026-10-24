#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
ti_pair=/workspace/benchmark-results/20261024_W4_QKVO64_fitted_fresh128_pair_tianshu
ti_c64=/workspace/benchmark-results/20261024_W4_QKVO64_fitted_v2_math64_tianshu
ti_c8=/workspace/benchmark-results/20261024_W4_QKVO64_fitted_v2_math8_tianshu
test ! -e "$ti_pair"
mkdir -p "$ti_pair/maintenance"
find experiments/tianshu_W4_QKVO64_fitted_fresh128_20261024 -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$ti_pair/controllers.sha256"
trap 'printf "screen_pipeline_exit_code=%s\n" "$?"' EXIT
printf 'waiting_for_quality_PID14085\n' > "$ti_pair/phase.txt"
while true; do
 ti_status=$(ps -p 14085 -o stat= || true)
 if test -z "$ti_status" || [[ "$ti_status" == Z* ]]; then break; fi
 sleep 20
done
for ti_root in "$ti_c64" "$ti_c8"; do
 if ! test -f "$ti_root/result.json" || ! jq -e '.phase=="complete" and .evaluation_exit_code==0 and .accuracy_passed==true and .total==105' "$ti_root/result.json" > /dev/null; then
  printf 'quality_not_qualified_no_GPU_screen\n' > "$ti_pair/phase.txt"
  printf 'Quality gate did not pass; no GPU performance service started.\n'
  exit 0
 fi
 jq -e '.complete==true and .fresh_entire_105==true and .all_caps_in_denominator==true and .above95==true' "$ti_root/accuracy_audit.json" > /dev/null
done
jq -e -s '.[0].frozen_sources==.[1].frozen_sources and .[0].server_command==.[1].server_command and (.[0].frozen_sources|length)==997' "$ti_c64/result.json" "$ti_c8/result.json" > /dev/null
ti_archive_status=$(ps -p 14473 -o stat= || true)
test -z "$ti_archive_status" || [[ "$ti_archive_status" == Z* ]]
grep -q '^storage=persistent_workspace$' experiments/tianshu_W4_QKVO64_fitted_results_20261024/SDK_RAW_ARCHIVE_32g.txt
for ti_arm in control candidate; do
 printf '%s_preparing\n' "$ti_arm" > "$ti_pair/phase.txt"
 ixsmi | grep -q 'No running processes found'
 ti_free_bytes=$(df --output=avail -B1 /workspace | tail -1)
 if test "$ti_free_bytes" -lt 1610612736 && test -d /tmp/torchinductor_root/fxgraph; then
  ti_archive_dir=$(mktemp -d /dev/shm/tianshu-joint-fxgraph-20261024.XXXXXX)
  find /tmp/torchinductor_root/fxgraph -type f -print0 | sort -z | xargs -0 -r sha256sum > "$ti_archive_dir/files.sha256"
  tar -C /tmp/torchinductor_root -cf "$ti_archive_dir/fxgraph_backup.tar" fxgraph
  tar -C /tmp/torchinductor_root -df "$ti_archive_dir/fxgraph_backup.tar"
  sha256sum -c "$ti_archive_dir/files.sha256" > "$ti_archive_dir/verification.log"
  sha256sum "$ti_archive_dir/fxgraph_backup.tar" > "$ti_archive_dir/archive.sha256"
  find /tmp/torchinductor_root/fxgraph -type f -delete
  cp "$ti_archive_dir/archive.sha256" "$ti_pair/maintenance/${ti_arm}_FX_archive.sha256"
 fi
 printf '%s_running\n' "$ti_arm" > "$ti_pair/phase.txt"
 python3 -u experiments/tianshu_W4_QKVO64_fitted_fresh128_20261024/run_model.py --output "$ti_pair/$ti_arm" --stage screen --batch 64 --mode "$ti_arm"
 jq -e '.phase=="complete" and .evaluation_exit_code==0' "$ti_pair/$ti_arm/result.json" > /dev/null
 printf '%s_screen_complete\n' "$ti_arm"
done
printf 'both_screens_complete_pending_comparison\n' > "$ti_pair/phase.txt"
sha256sum -c "$ti_pair/controllers.sha256" > "$ti_pair/controllers_verified.txt"
python3 experiments/tianshu_W4_QKVO64_fitted_fresh128_20261024/compare_screens_cpu.py
printf 'joint_fresh128_complete_pending_review\n' > "$ti_pair/phase.txt"
