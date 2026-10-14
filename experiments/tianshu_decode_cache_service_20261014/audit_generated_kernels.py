# SPDX-License-Identifier: Apache-2.0
"""Verify archived actual .cv kernel bytes and their original L0 provenance."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    state = json.loads((root / "result.json").read_text())
    assert state["phase"] in ("complete", "evaluation_finished_pending_report_audit")
    flags = dict(state.get("environment_flags", {}), **state.get("actual_composed_environment", {}))
    remote_root = Path(flags["TIANSHU_DECODE_COHERENT_RECORD_DIR"]).parent
    helper = Path(__file__).resolve().parents[1] / "tianshu_decode_cache_policy_20261014/experimental_cache.py"
    spec = importlib.util.spec_from_file_location("_cv_archive_renderer", helper)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    routes = [json.loads(path.read_text()) for path in (root / "decode_coherent_records").glob("producer_*.json")]
    expected = state.get("decode_coherent_records", state.get("candidate_routing_records"))
    assert routes and expected
    key = lambda item: (item["batch"], item["segments"])
    assert sorted(routes, key=key) == sorted(expected, key=key)
    checked = {}
    for route in routes:
        assert route["cache_service_mode"] == "candidate" and route["actual_cache_modifier"] == ".cv"
        assert route["cache_helper_sha256"] == sha(helper)
        provenance = route["zero_initialization"]
        original = provenance["original_L0_generation"]
        cache = route["cache_policy_generation"]
        paths = dict(actual=Path(provenance["generated_source"]),
                     base_L0=Path(original["generated_source"]),
                     cache=Path(cache["generated_source"]))
        local = {name: root / path.relative_to(remote_root) for name, path in paths.items()}
        assert local["actual"] == local["cache"]
        assert sha(local["actual"]) == route["generated_source_sha256"] == "8deb8967fdacf354f3b38e6a9687163493f6163764f3e8663dce916250cd1289"
        assert sha(local["base_L0"]) == original["generated_sha256"] == cache["original_L0_sha256"] == "4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e"
        assert renderer.render_source(local["base_L0"].read_bytes()).encode() == local["actual"].read_bytes()
        assert original["original_source_sha256"] == "bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0"
        for path in local.values(): checked[str(path.relative_to(root))] = sha(path)
    output = dict(complete=True, actual_routes_verified=len(routes), generated_files=checked,
                  actual_cv_bytes_reproduced_from_original_L0=True, provenance_chain_native_to_L0_to_cv_verified=True,
                  numeric_or_performance_claim=False, helper_sha256=sha(helper))
    (root / "local_generated_kernel_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__": main()
