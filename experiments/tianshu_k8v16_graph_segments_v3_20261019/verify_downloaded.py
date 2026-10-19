# SPDX-License-Identifier: Apache-2.0
"""Recompute actual replay links and validate every diagnostic response from downloaded artifacts."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from graph_audit import audit_trace

base = Path(__file__).resolve().parent
repo = base.parents[1]
p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


state = read(root/'result.json')
assert state['phase'] == 'complete' and not state['official_benchmark'] and not state['adopted']
assert state['instrumented_diagnostic_not_a_performance_score']
for name, sha in state['frozen_sources'].items():
    assert digest(root/'source_snapshot'/Path(name).relative_to('/')) == sha
    if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
        assert digest(repo/Path(name).relative_to('/workspace/vllm-plugin-FL')) == sha
spec = importlib.util.spec_from_file_location('_graph_segments_layout_download_audit',base/'layout_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
profile = audit.audit_runtime(root,state,official=True,online=False)
assert profile == state['profile_audit'] and profile['K8V16'] and profile['zero_preemptions']
graph = audit_trace(root,state)
assert graph == state['graph_audit'] == read(root/'graph_audit.json')
reference = state['canary_reference']
refpath = root/'source_snapshot'/Path(reference['path']).relative_to('/')
assert digest(refpath) == reference['sha256'] == state['frozen_sources'][reference['path']]
golden = read(refpath)['seeded_canaries'][0]['text']
assert len(state['seeded_canaries']) == 3
for response in state['seeded_canaries']:
    assert response['text'] == golden and response['api_seed'] == 777
    assert response['usage'] == dict(prompt_tokens=128,completion_tokens=16,total_tokens=144)
assert len(state['diagnostic_workloads']) == 6
responses = 0
for index,(entry,count) in enumerate(zip(state['diagnostic_workloads'],(1,8,16,24,32,64))):
    assert entry['concurrency'] == entry['successful_requests'] == len(entry['responses']) == count
    assert entry['input_tokens'] == 4096 and entry['output_tokens'] == 128
    for j,response in enumerate(entry['responses']):
        assert response['input_seed'] == 202610193000+100*index+j
        assert response['usage'] == dict(prompt_tokens=4096,completion_tokens=128,total_tokens=4224)
        assert response['wall_seconds'] >= response['ttft_seconds'] > 0
        responses += 1
assert responses == 145
result = dict(complete=True,diagnostic_only=True,adopted=False,sources=len(state['frozen_sources']),
    environment_keys=43,all145_load_responses_and3_reference_canaries_checked=True,
    actual_Graph_capture_configuration_to_long_context_replay_links=graph['actual_capture_graph_object_replay_links'],
    no_performance_or_new_quality_claim=True)
(root/'local_download_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
