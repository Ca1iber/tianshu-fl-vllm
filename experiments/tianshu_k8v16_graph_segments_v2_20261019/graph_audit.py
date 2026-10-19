# SPDX-License-Identifier: Apache-2.0
"""Join actual graph-object replay edges to native kernel launches observed inside capture."""
import json
from pathlib import Path


def audit_trace(root, state):
    paths = list((root/'graph_segments_records').glob('trace_*.json'))
    workers = []
    for path in paths:
        record = json.loads(path.read_text())
        if set(record['bound']) == {'graph', 'runner', 'routing'}:
            workers.append(record)
    assert len(workers) == 1
    trace = workers[0]
    assert trace['diagnostic_only'] and trace['attention_arguments_and_return_values_forwarded']
    assert trace['RNG_not_read_or_modified']
    for path, sha in trace['sources'].items():
        assert state['frozen_sources'][path] == sha
    graphs, launches = trace['graphs'], trace['launches']
    assert graphs and launches
    links = []
    for key, graph in graphs.items():
        assert graph['runtime_mode'] == 'FULL' and graph['capture_object_id'] is not None
        assert graph['capture_calls'] == 1
        segments=graph['segments']
        assert segments and segments['GPU_graphs']>=1
        assert len(segments['entries'])==segments['GPU_graphs']+segments['eager_breaks']
        ids=[x['graph_object_id'] for x in segments['entries'] if x['kind']=='GPU_graph']
        assert len(set(ids))==segments['GPU_graphs'] and all(isinstance(x,int) and x>0 for x in ids)
        captured = [item for item in launches.values()
            if item['graph_key'] == key and item['wrapper_action'] == 'capture']
        assert captured
        configurations = {json.dumps(item['configuration'], sort_keys=True) for item in captured}
        assert len(configurations) == 1
        config = captured[0]['configuration']
        assert sum(item['calls'] for item in captured) == 42
        if config['IS_3D']:
            assert config['tile'] == 16 and config['warps'] == config['stages'] == 1
            assert 1 <= config['sequences'] <= 64
        for nreq, frame in graph['frames'].items():
            if frame['max_computed'] < 4096:
                continue
            assert graph['replay_calls'] >= frame['count'] > 0
            links.append(dict(requests=int(nreq), descriptor=graph['descriptor'],
                capture_object_id=graph['capture_object_id'], graph_replay_calls=graph['replay_calls'], breakable_segments=segments,
                native_calls_inside_GPU_capture=sum(x['calls'] for x in captured if x['actually_inside_capture']),
                native_calls_eager_during_capture=sum(x['calls'] for x in captured if not x['actually_inside_capture']),
                eager_replay_configurations=[x['configuration'] for x in launches.values() if x['graph_key']==key and x['wrapper_action']=='replay'],
                observed_frame_count=frame['count'], min_computed=frame['min_computed'],
                max_computed=frame['max_computed'], configuration=config,
                capture_min_max_seqlen_k=min(item['min_max_seqlen_k'] for item in captured),
                capture_max_max_seqlen_k=max(item['max_max_seqlen_k'] for item in captured)))
    assert {1, 8, 16, 24, 32, 64} <= {row['requests'] for row in links}
    return dict(complete=True, diagnostic_only=True, adopted=False,
        actual_capture_graph_object_replay_links=links, graph_count=len(graphs),
        all_capture_edges_have42_native_layers=True, native_layers_inside_GPU_capture_or_eager_break_explicitly_recorded=True,
        counters_are_periodic_or_exit_snapshot_not_claimed_complete_step_trace=True,
        source_hashes_bound_to_frozen_snapshots=True, performance_score_not_asserted=True,
        model_quality_not_retested=True)
