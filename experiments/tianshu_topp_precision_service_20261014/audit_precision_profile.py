# SPDX-License-Identifier: Apache-2.0
"""Verify actual candidate worker configuration and generated TopP source bytes."""
import hashlib
import json
from pathlib import Path


def audit_precision_profile(root):
    root = Path(root)
    top = [json.loads(p.read_text()) for p in (root / "topp_launch_records").glob("route_*.json")]
    worker = [json.loads(p.read_text()) for p in (root / "max_seqs_records").glob("worker_*.json")]
    assert top and worker
    for record in worker:
        assert record["mode"] == "candidate" and record["max_num_seqs"] == 64
        assert record["max_num_batched_tokens"] == 4096 and record["model_max_len"] == 131072
        assert record["async_scheduling"]
    pure = [r for r in top if r["topp"] and not r["topk"]]
    assert pure and any(r["batch"] == 8 for r in pure)
    generated_files = set()
    for record in top:
        active = bool(record["topp"] and not record["topk"])
        assert record["precision_service_mode"] == "candidate" and record["corrected_source_active"] == active
        assert record["actual_BLOCK_SIZE"] == (16384 if active else 8192)
        assert record["actual_BLOCK_SIZE_TRUNC"] == (8192 if active else 4096)
        assert record["configuration"] == dict(num_warps=16, num_stages=1)
        assert record["compiled_metadata"]["num_warps"] == 16 and record["compiled_metadata"]["num_stages"] == 1
        assert record["sampling_parameters_and_RNG_unchanged"]
        if active:
            assert record["actual_source_sha256"] == "029d805b1a19be91f74afa03afb25c834409c8f08b3d30f7eabd1771c4aefedf"
            generation = record["generation"]
            path = root / "topp_launch_records" / Path(generation["generated_source"]).parent.name / "generated_topp_precision.py"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == generation["generated_sha256"] == record["actual_source_sha256"]
            assert generation["native_TopK_branch_AST_unchanged"] and generation["reversible_to_exact_native_source"]
            generated_files.add(str(path))
        else:
            assert record["actual_source_sha256"] == "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8"
    return dict(complete=True, combined_TopP_precision_and_maxseq64=True,
                actual_max_num_seqs=64, generated_source_files=sorted(generated_files),
                actual_worker_records=worker, actual_TopP_records=top)
