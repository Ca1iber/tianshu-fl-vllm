# SPDX-License-Identifier: Apache-2.0
"""Check diagnostic attribution and pass-through using fake runner/graph calls."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from route_observer import install


def main():
    modes = {name: NS(name=name) for name in ('NONE', 'FULL')}
    current = [None]

    class Graph:
        def __init__(self):
            self.runtime_mode = modes['FULL']
            self.concrete_graph_entries = {'known': NS(graph=object())}
            self.native_calls = 0

        def __call__(self, value):
            self.native_calls += 1
            return value * 7

    graph = Graph()

    class Runner:
        uniform_decode_query_len = 2
        input_batch = NS(num_computed_tokens_cpu=[4096] * 3, num_prompt_tokens=[4096] * 3)
        original_dispatches = 0

        def _determine_batch_execution_and_padding(self, num_tokens, num_reqs,
                num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn,
                force_uniform_decode=None, force_eager=False):
            self.original_dispatches += 1
            mode = modes['FULL'] if max_num_scheduled_tokens == 2 and num_tokens == 2 * num_reqs else modes['NONE']
            return mode, NS(num_tokens=num_tokens, num_reqs=num_reqs, uniform=mode.name == 'FULL')

        def execute_model(self, output, fail=False):
            values = list(output.num_scheduled_tokens.values())
            mode, desc = self._determine_batch_execution_and_padding(sum(values), len(values), values, max(values), False)
            current[0] = NS(cudagraph_runtime_mode=mode, batch_descriptor='known')
            if fail:
                raise ValueError('preserve original exception')
            return graph(13)

    owner = Runner()
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        marker = root / 'phase'
        marker.write_text('warmup')
        recorder = install(Runner, Graph, lambda: current[0], root / 'records', marker)
        outside = owner._determine_batch_execution_and_padding(2, 1, [2], 2, False)
        assert outside[0] is modes['FULL']
        variants = ([1, 1, 1], [2, 2, 2], [1, 2, 1])
        for values in variants:
            out = NS(num_scheduled_tokens=dict(enumerate(values)), scheduled_spec_decode_tokens={1: [9]})
            assert owner.execute_model(out) == 91
        marker.write_text('measurement')
        owner.input_batch.num_computed_tokens_cpu = [0, 4096, 4096]
        out = NS(num_scheduled_tokens={0: 2, 1: 1, 2: 1}, scheduled_spec_decode_tokens={0: [5]})
        assert owner.execute_model(out) == 91
        try:
            owner.execute_model(out, fail=True)
        except ValueError as error:
            assert str(error) == 'preserve original exception'
        else:
            raise AssertionError('Exception swallowed')
        assert recorder.local.frame is None
        recorder.dump()
        data = json.loads(next((root / 'records').glob('coverage_*.json')).read_text())
        assert data['calls'] == 5 and sum(row['count'] for row in data['rows']) == 5
        assert data['outside_execute'] == {'dispatch:FULL': 1}
        rows = [r['frame'] for r in data['rows']]
        assert [r['label'] for r in rows].count('warmup') == 3
        assert rows[0]['dispatches'][0]['mode'] == 'NONE'
        assert rows[1]['graph_calls'][0]['branch'] == 'existing_graph_replay'
        assert rows[2]['graph_calls'][0]['branch'] == 'eager_passthrough'
        assert rows[3]['dispatches'][0]['stage'] == 'prefill_or_mixed'
        assert len(data['errors']) == 1 and len(rows[4]['graph_calls']) == 0
        assert graph.native_calls == 4 and owner.original_dispatches == 6
    print(json.dumps(dict(complete=True, fake_execute_cases=5, outside_dispatch_cases=1,
                          original_returns_and_exception_preserved=True, GPU_test=False)))


if __name__ == '__main__':
    main()
