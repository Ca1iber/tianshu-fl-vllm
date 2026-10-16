# SPDX-License-Identifier: Apache-2.0
"""Exercise actual controller qualification guards without reaching any GPU calls."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile


def main():
    base = Path(__file__).resolve().parent
    source = base / "run_service_pair.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "main")
    relatives = []
    for n in function.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == "relatives" for t in n.targets):
            relatives.extend(ast.literal_eval(n.value))
        if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id == "relatives":
            relatives.extend(ast.literal_eval(n.value))
    assert len(relatives) == len(set(relatives)) == 23
    frozen = {str(base.parent / p):hashlib.sha256((base.parent / p).read_bytes()).hexdigest() for p in relatives}
    boundary = next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assert)
        and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Constant)
        and n.test.left.value == "No running processes found")
    function.body = function.body[:boundary] + [ast.Return(value=ast.Name(id="excluded",ctx=ast.Load()))]
    scope = dict(argparse=argparse,Path=Path,json=json,hashlib=hashlib)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(source),"exec"),scope)
    results = []
    with tempfile.TemporaryDirectory(prefix="tianshu_combo_gate_") as temp:
        root = Path(temp)
        quality = root / "quality";quality.mkdir()
        official = root / "official";official.mkdir()
        screen = root / "screen";screen.mkdir()
        (quality / "accuracy_audit.json").write_text(json.dumps(dict(complete=True,total=105,accuracy=103/105,fresh_math_for_decode_cache_cv=True)))
        (official / "comparison.json").write_text(json.dumps(dict(fresh_math_for_decode_cache_cv=True,adopt_as_best=True,requests_successful=1024)))
        for case in ("three_round","five_round_ok","missing_screen","incomplete_screen","small_gain","overlap","source_changed"):
            proof = dict(complete=True,successful_requests=384,excluded_rounds=[1],gain_percent=2.5,
                control_rounds=[3900,4050,4051],candidate_rounds=[3900,4140,4141],frozen_sources=dict(frozen))
            state = dict(phase="failed" if case == "incomplete_screen" else "complete")
            if case == "small_gain": proof["gain_percent"] = 1.9
            if case == "overlap": proof["candidate_rounds"][1] = 4049
            if case == "source_changed": proof["frozen_sources"][next(iter(frozen))] = "0" * 64
            (screen / "comparison.json").write_text(json.dumps(proof))
            (screen / "result.json").write_text(json.dumps(state))
            rounds = 3 if case == "three_round" else 5
            sys.argv = [str(source),"--output",str(root/"unused"),"--quality-root",str(quality),"--official-root",str(official),"--rounds",str(rounds)]
            if case != "missing_screen": sys.argv += ["--screen-root",str(screen)]
            expected = case in ("three_round","five_round_ok")
            try:
                excluded = scope["main"]()
            except AssertionError:
                assert not expected
            else:
                assert expected and excluded == ([1] if rounds == 3 else [1,2])
            results.append(dict(case=case,expected_pass=expected))
    result = dict(complete=True,cpu_only=True,actual_controller_guards_executed=True,
        source23_exist_and_unique=True,no_GPU_code_reached=True,runs=results,
        controller_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    out = base / "results/20261016/prepared_32g";out.mkdir(parents=True,exist_ok=True)
    (out/"CPU_CONTROLLER_GATE_AUDIT_32g.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__": main()
