#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cache_gate="$base/../tianshu_prefill_control_binary_all_20261022/QUALIFIED_MODEL_CACHE_32g.json"
test "$#" -le 1
mode=${1:-restore};case "$mode" in restore|--check);;*)printf 'Usage: %s [--check]\n' "$0" >&2;exit 2;;esac
sha256sum --check --status "$base/QUALIFIED_RELEASE.sha256"
jq -e '.complete and .total_strict_binary_points==24 and (.files|length)>0' "$cache_gate" >/dev/null
mapfile -d '' -t pairs < <(jq -j '.files|to_entries[]|.value,"\u0000",.key,"\u0000"' "$cache_gate")
declare -A available=()
missing=0;restored=0;checked=0
for ((i=0;i<${#pairs[@]};i+=2));do
    expected=${pairs[i]};filename=${pairs[i+1]}
    if [[ -f "$filename" ]];then
        actual=$(sha256sum < "$filename");actual=${actual%% *}
        [[ "$actual" == "$expected" ]] || { printf 'Existing frozen cache differs: %s\n' "$filename" >&2;exit 1; }
        available["$expected"]=$filename;checked=$((checked+1))
    fi
done
for ((i=0;i<${#pairs[@]};i+=2));do
    expected=${pairs[i]};filename=${pairs[i+1]}
    [[ -f "$filename" ]] && continue
    copy_from=${available[$expected]-}
    if [[ "$mode" == --check || -z "$copy_from" ]];then
        printf 'Missing frozen cache: %s\n' "$filename" >&2;missing=$((missing+1));continue
    fi
    mkdir -p -- "$(dirname -- "$filename")"
    cp -- "$copy_from" "$filename"
    actual=$(sha256sum < "$filename");actual=${actual%% *};test "$actual" = "$expected"
    restored=$((restored+1))
done
printf 'Frozen cache files checked=%s restored=%s missing=%s\n' "$checked" "$restored" "$missing"
test "$missing" -eq 0
