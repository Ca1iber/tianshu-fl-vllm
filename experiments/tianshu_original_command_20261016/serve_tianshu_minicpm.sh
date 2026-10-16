#!/usr/bin/env bash
set -euo pipefail

# Original official serve arguments with the measured operator profile.
# See results/20261016 for validation status; no score is implied by this launcher.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env -u PYTHONPATH python3 - "${script_dir}" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

base = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(base))
from validation_profile import official_command

template = json.loads((base / "profile_environment_template.json").read_text())
dry_run = os.environ.get("TIANSHU_ORIGINAL_PROFILE_DRY_RUN") == "1"
root = Path(os.environ.get("TIANSHU_ORIGINAL_RECORD_DIR", "/workspace/benchmark-results/original_command_" +
    datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(os.getpid()))).resolve()
old = template["old_output_root"]
profile = {key: value.replace(old + "/", str(root) + "/") for key, value in template["environment"].items()}
assert len(profile) == 35 and all(old not in value for value in profile.values())
environment = dict(os.environ)
for key in list(environment):
    if key.startswith(("TIANSHU_", "VLLM_FL_", "FLAGGEMS_ILUVATAR_")) or key in {
        "VLLM_PLUGINS", "PYTHONPATH", "PYTHONUNBUFFERED", "TRITON_PRINT_AUTOTUNING", "CUDA_VISIBLE_DEVICES"
    }:
        del environment[key]
environment.update(profile)
command = official_command()
if dry_run:
    print(json.dumps(dict(command=command, profile_environment=profile)))
else:
    root.mkdir(parents=True, exist_ok=False)
    (root / "launcher_profile.json").write_text(json.dumps(dict(command=command,
        profile_environment=profile), indent=2) + "\n")
    os.execvpe(command[0], command, environment)
PY
