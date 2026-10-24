#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
ti_batch=${1:?batch 8 or 64 required}
case "$ti_batch" in 8|64);;*)exit 2;;esac
ti_results=experiments/tianshu_r5_BM8_capacity_results_20261024
ti_root=/dev/shm/tianshu_r5_BM8_capacity_runs_20261024/math${ti_batch}
ti_archive=/workspace/benchmark-results/20261024_BM8_capacity_math${ti_batch}_raw.tar.xz
jq -e '.phase=="complete" and .evaluation_exit_code==0 and .accuracy_passed and .total==105' "$ti_root/result.json" > /dev/null
jq -e '.complete and .above95 and .fresh_entire_105 and .all_caps_in_denominator and .total==105' "$ti_root/accuracy_audit.json" > /dev/null
jq -e '.complete and .native_capacity_quantized and .physical_KV_blocks==36096 and .additional_payload_bytes==210567168' "$ti_root/profile_audit.json" > /dev/null
mapfile -d '' ti_report < <(find "$ti_root/evalscope" -path '*/reports/minicpm/math_500.json' -print0)
test "${#ti_report[@]}" = 1
ti_run=${ti_report[0]%/reports/minicpm/math_500.json}
ti_review="$ti_run/reviews/minicpm/math_500_Level 3.jsonl"
ti_pred="$ti_run/predictions/minicpm/math_500_Level 3.jsonl"
jq -n -r --slurpfile r "$ti_review" --slurpfile p "$ti_pred" '
  if ($r|length)!=105 or ($p|length)!=105 then error("Incomplete dataset") else
  ["index","accuracy","stop_reason","error"],
  ($r|sort_by(.index)[] as $row |
   $p[]|select(.index==$row.index)|
   if $row.sample_score.score.status!="success" or .model_output.error!=null then error("Request failure") else
   [$row.index,$row.sample_score.score.value.accuracy,.model_output.choices[0].stop_reason,false] end) end | @csv' > "$ti_results/C${ti_batch}_KEY_SCORES_32g.csv"
jq -s --argjson batch "$ti_batch" --arg root "$ti_root" '{complete:true,batch:$batch,raw_root:$root,
 correct:.[1].correct,total:105,accuracy_percent:(100*.[1].correct/105),above95:.[1].above95,
 all105_and_caps_counted:true,request_errors:0,stop_reasons:.[1].stop_reasons,incorrect_indices:.[1].incorrect_indices,
 original_serve_argv:.[0].server_command,source_count:(.[0].frozen_sources|length),environment_count:(.[0].full_environment|length),
 physical_KV_blocks:.[2].physical_KV_blocks,native_capacity_proof:.[2].capacity_proof,
 actual_FULL64_replays:.[2].actual_FULL64_replays,additional_payload_bytes:.[2].additional_payload_bytes,
 original997_exact_subset:.[2].original997_exact_subset,official_benchmark:false,adopted:false}' \
 "$ti_root/result.json" "$ti_root/accuracy_audit.json" "$ti_root/profile_audit.json" > "$ti_results/C${ti_batch}_QUALITY_GATE_32g.json"
sha256sum "$ti_root/result.json" "$ti_root/accuracy_audit.json" "$ti_root/profile_audit.json" "${ti_report[0]}" "$ti_review" "$ti_pred" > "$ti_results/C${ti_batch}_RAW.sha256"
if test ! -e "$ti_archive"; then
 tar -C "$ti_root" --exclude=./source_snapshot --exclude=./vllm_cache -cJf "$ti_archive" .
fi
tar -C "$ti_root" -dJf "$ti_archive"
tar -xOJf "$ti_archive" ./result.json | cmp - "$ti_root/result.json"
sha256sum "$ti_archive" > "$ti_results/C${ti_batch}_PERSISTENT_ARCHIVE.sha256"
jq '{batch,correct,total,accuracy_percent,above95,physical_KV_blocks}' "$ti_results/C${ti_batch}_QUALITY_GATE_32g.json"
