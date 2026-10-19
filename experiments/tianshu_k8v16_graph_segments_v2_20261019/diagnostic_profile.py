# SPDX-License-Identifier: Apache-2.0
"""Preserve the qualified K8V16 profile; add only bounded, non-numeric instrumentation."""
import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_graph_segments_qualified_profile',
    BASE.parent/'tianshu_k8v16_pipeline_20261019/candidate_profile.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
digest, frozen_check, read = parent.digest, parent.frozen_check, parent.read


def prepare(root, mode):
    assert mode == 'candidate'
    gate = read(BASE/'CPU_GATE_32g.json')
    assert gate['complete'] and gate['actual_installed_Graph_call_AST_executed'] and gate['actual_Breakable_wrapper_and_capture_AST_executed']
    for name, sha in gate['sources'].items():
        assert digest(BASE/name) == sha, name
    command, environment, profile, sources = parent.prepare(root, 'candidate')
    assert len(profile) == 41
    profile.update(PYTHONPATH=str(BASE/'hooks'), TIANSHU_GRAPH_SEGMENTS_PROBE='1',
        TIANSHU_GRAPH_SEGMENTS_RECORD_DIR=str(root/'graph_segments_records'))
    assert len(profile) == 43
    environment.update(profile)
    graph = Path('/usr/local/lib/python3.12/site-packages/vllm/compilation/breakable_cudagraph.py')
    assert digest(graph) == '722683b44b4461ffd9a6498741da18945cbaac231c3bb01f0b0443dafe9e053e'
    paths = [BASE/name for name in ('diagnostic_profile.py', 'layout_audit.py', 'trace.py',
        'hooks/sitecustomize.py', 'run_diagnostic.py', 'graph_audit.py', 'check_cpu.py', 'CPU_GATE_32g.json')]
    paths.append(graph)
    reference = Path('/workspace/benchmark-results/20261019_k8v16_engine_service_tianshu/candidate/result.json')
    assert read(reference)['phase'] == 'complete'
    paths.append(reference)
    sources.update({str(p):digest(p) for p in paths})
    frozen_check(sources)
    for path, sha in sources.items():
        destination = root/'source_snapshot'/Path(path).relative_to('/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(path).read_bytes())
        assert digest(destination) == sha
    return command, environment, profile, sources
