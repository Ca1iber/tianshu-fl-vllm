# SPDX-License-Identifier: Apache-2.0
"""Recompute timings and verify all frozen sources of the sort probe."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    base = Path(__file__).resolve().parent
    state = json.loads((root / "result.json").read_text())
    assert state["phase"] == "complete"
    assert not state["official_benchmark"] and not state["installed_operator_modified"]
    assert state["cases"] == 13 and state["variants"] == 6 and state["rows"] == 78
    assert state["numerical_failures"] == 0 and state["changed_input_graph_checks"] == 78
    assert state["events"] == 936
    assert len(state["frozen_sources"]) == 8
    for source, digest in state["frozen_sources"].items():
        remote = Path(source)
        if remote.is_relative_to("/workspace/vllm-plugin-FL"):
            local = repo / remote.relative_to("/workspace/vllm-plugin-FL")
            snapshot = root / "source_snapshot" / local.relative_to(base)
            assert hashlib.sha256(local.read_bytes()).hexdigest() == digest
            assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest
        else:
            assert str(remote) == "/usr/local/lib/python3.12/site-packages/flag_gems/ops/sort.py"
            assert digest == hashlib.sha256((base / "baseline_sort.py").read_bytes()).hexdigest()
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 78
    cases = {}
    for row in rows:
        assert row["values_bitwise_equal"] and row["stable_indices_int64_equal"]
        assert row["changed_input_graph_equal"]
        assert row["input_sha256"] != row["changed_input_sha256"]
        assert len(row["event_blocks"]) == 3
        assert all(len(block) == 4 for block in row["event_blocks"])
        samples = [value for block in row["event_blocks"] for value in block]
        assert all(value > 0 for value in samples)
        assert abs(statistics.median(samples) - row["median_ms"]) < 1e-9
        group = cases.setdefault(row["case_id"], {})
        assert row["variant"] not in group
        group[row["variant"]] = row
    comparisons = []
    pairs = (("installed_baseline", "empty_bin_only"),
             ("installed_baseline", "metax_int32_reference"),
             ("metax_int32_reference", "metax_empty_bin"),
             ("metax_empty_bin", "metax_two_bin"),
             ("metax_two_bin", "metax_repeat_digit"),
             ("installed_baseline", "metax_repeat_digit"))
    for case_id, group in sorted(cases.items()):
        assert len(group) == 6
        assert len({row["input_sha256"] for row in group.values()}) == 1
        assert len({row["changed_input_sha256"] for row in group.values()}) == 1
        for control, candidate in pairs:
            aa, bb = group[control], group[candidate]
            comparisons.append(dict(case_id=case_id, batch=aa["batch"], distribution=aa["distribution"],
                descending=aa["descending"], control=control, candidate=candidate,
                control_ms=aa["median_ms"], candidate_ms=bb["median_ms"],
                time_change_percent=100 * (bb["median_ms"] / aa["median_ms"] - 1)))
    assert len(cases) == 13
    output = dict(complete=True, numerical_checks=78, changed_input_graph_checks=78,
                  recomputed_device_events=936, frozen_source8_verified=True,
                  service_installed=False, service_throughput_claim=False,
                  B64_generic_sort_is_operator_control_only=True, comparisons=comparisons)
    (root / "local_download_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "comparisons"}))
    for row in comparisons:
        if row["descending"] and row["distribution"] == "finite_fp32":
            print(json.dumps(row))


if __name__ == "__main__":
    main()
