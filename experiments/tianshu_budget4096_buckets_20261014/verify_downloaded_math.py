# SPDX-License-Identifier: Apache-2.0
"""Verify new B64S8 Math105 with five-source namespaced snapshots and five-round proof."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from audit_profile import audit_profile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--confirm-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    confirm = args.confirm_root.resolve()
    repo = Path(__file__).resolve().parents[2]
    state = json.loads((root / "result.json").read_text())
    audit = json.loads((root / "accuracy_audit.json").read_text())
    trial = json.loads((confirm / "candidate/result.json").read_text())
    proof = json.loads((confirm / "comparison.json").read_text())
    assert state["phase"] == "evaluation_finished_pending_report_audit"
    assert state["evalscope_exit_code"] == 0
    assert proof["complete"] and proof["gain_percent"] >= 2
    assert min(proof["candidate_rounds"][2:]) > max(proof["control_rounds"][2:])
    assert audit["complete"] and audit["total"] == 105
    assert not audit["reused_original_accuracy"]
    assert audit["fresh_math_for_B64S8_segments"]
    assert Path(audit["matched_service_root"]).name == confirm.name
    assert state["server_command"] == audit["server_command"] == trial["command"]
    assert state["installed_sources"] == audit["installed_sources"] == trial["installed_sources"]
    assert state["candidate_hook"]["sha256"] == trial["active_hook"]["sha256"]
    assert len(audit["segment_sources"]) == 10
    for path, digest in audit["segment_sources"].items():
        relative = Path(path).relative_to("/workspace/vllm-plugin-FL")
        assert sha(repo / relative) == digest
        assert sha(root / "source_snapshot/segment_sources" / relative.relative_to("experiments")) == digest
    assert state["segment_sources"] == audit["segment_sources"]
    for name, meta in audit["installed_sources"].items():
        assert sha(root / "source_snapshot" / name) == meta["sha256"]
    reports = list(root.glob("evalscope/*/reports/minicpm/math_500.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    run = reports[0].parents[2]
    predictions = [json.loads(x) for x in (run / "predictions/minicpm/math_500_Level 3.jsonl").read_text().splitlines()]
    reviews = [json.loads(x) for x in (run / "reviews/minicpm/math_500_Level 3.jsonl").read_text().splitlines()]
    assert len(predictions) == len(reviews) == 105
    assert len({x["index"] for x in predictions}) == len({x["index"] for x in reviews}) == 105
    assert {x["index"] for x in predictions} == {x["index"] for x in reviews}
    assert all(x["model_output"]["error"] is None for x in predictions)
    execution = report["execution_summary"]
    assert execution["requested"] == execution["succeeded"] == report["num"] == 105
    assert execution["errored"] == 0 and execution["incomplete"] is False
    assert all(x["sample_score"]["score"]["status"] == "success" for x in reviews)
    values = [x["sample_score"]["score"]["value"]["accuracy"] for x in reviews]
    assert all(x in (0, 1) for x in values)
    correct = int(sum(values))
    assert correct == audit["correct"]
    assert correct / 105 == audit["accuracy"]
    assert audit["official_requirement_passed"] == (correct / 105 > .95)
    stops = dict(Counter(x["model_output"]["choices"][0]["stop_reason"] for x in predictions))
    assert stops == audit["generation_stop_reasons"]
    for path, meta in audit["raw_files"].items():
        p = root / path
        assert p.stat().st_size == meta["bytes"] and sha(p) == meta["sha256"]
    command = audit["evaluation_command"]
    assert command[command.index("--eval-batch-size") + 1] == "8"
    assert command[command.index("--datasets") + 1] == "math_500"
    assert command[command.index("--eval-type") + 1] == "openai_api"
    assert json.loads(command[command.index("--generation-config") + 1]) == dict(temperature=1., top_p=.95, max_tokens=32768)
    dataset = json.loads(command[command.index("--dataset-args") + 1])
    assert dataset == {"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}
    routes = audit["candidate_routing_records"]
    assert routes
    for route in routes:
        tile = 16 if route["batch"] in (8,16,32,40,48,56,64) else 32
        warps = 1 if tile == 16 else 2 if 8 <= route["batch"] <= 64 else 4
        assert route["configuration"] == dict(TILE_SIZE=tile, num_warps=warps, num_stages=1)
        assert route["compiled_metadata"]["num_warps"] == warps
        assert route["generated_source_sha256"] == "4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e"
    assert any(route["batch"] == 8 and route["configuration"] == dict(TILE_SIZE=16,num_warps=1,num_stages=1) for route in routes)
    policy=audit['segment_policy_records']
    assert policy and any(x['batch']==64 and x['selected_segments']==8 and x['baseline_segments']==4 for x in policy)
    assert all(x['selected_segments']==(8 if x['batch']==64 else x['baseline_segments']) for x in policy)
    assert any(x['batch']==64 and x['segments']==8 for x in routes)
    for x in routes:
        expected=8 if x['batch']==64 else 64 if x['batch']==1 else 4 if x['batch']>=16 else 16
        assert x['segments']==expected
    assert 'official Math concurrency8' in audit['B64_binding_scope']
    reducers = audit["decode_coherent_reducer_records"]
    assert {(x["batch"], x["segments"]) for x in routes} == {(x["batch"], x["segments"]) for x in reducers}
    assert reducers and all(x["actual_tile"] == (16 if x["batch"] in (8,16,32,40,48,56,64) else 32) for x in reducers)
    topp = audit["topp_launch_records"]
    assert topp and all(x["configuration"] == dict(num_warps=16, num_stages=1) and x["compiled_metadata"]["num_warps"] == 16 for x in topp)
    traced = root / "bucket_trace_records"
    warmed = [json.loads(p.read_text()) for p in traced.glob("warmup_completed_*.json")]
    assert any(x["complete"] and len(x["calls"]) == 83 for x in warmed)
    gather = [json.loads(p.read_text()) for p in traced.glob("gather_*.json")]
    assert gather and any(x["index_dtype"] == "torch.int32" for x in gather)
    assert "No running processes found" in (root / "hardware_after.txt").read_text()
    profile = audit_profile(root,state)
    assert profile == audit["budget4096_profile"]
    result = dict(budget4096_profile=profile,complete=True, correct=correct, total=105, accuracy=correct/105,
                  official_requirement_passed=correct/105>.95, prediction_errors=0,
                  generation_stop_reasons=stops, sources10_verified=True,
                  installed_sources6_verified=True, actual_cli_and_eval_params_verified=True,
                  actual_guarded_KV16_w1_B64S8_topp16_and83warm_verified=True, raw_predictions_reviews_verified=105)
    (root / "local_download_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
