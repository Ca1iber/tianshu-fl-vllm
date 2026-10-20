#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:-/workspace/benchmark-results}
case "$root" in /*) ;; *) printf 'Expected an absolute destination directory\n' >&2; exit 2;; esac
# Only copy the three frozen Python inputs, to the 22 paths in the manifest.
# Existing files must already match; a different file is never overwritten.
while IFS=$'\t' read -r digest relative; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ && "$relative" != /* && "$relative" != *..* && "$relative" == *.py ]]
    source="$base/sources/$digest.py"
    destination="$root/$relative"
    printf '%s  %s\n' "$digest" "$source" | sha256sum --check --status
    if test -e "$destination" || test -L "$destination"; then
        printf '%s  %s\n' "$digest" "$destination" | sha256sum --check --status
    else
        mkdir -p -- "$(dirname -- "$destination")"
        cp -- "$source" "$destination"
    fi
    printf '%s  %s\n' "$digest" "$destination" | sha256sum --check --status
done < "$base/restore_paths.tsv"
printf 'Verified 22 historical Python inputs in %s\n' "$root"
