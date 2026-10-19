# SPDX-License-Identifier: Apache-2.0
"""Observe real execute/dispatch/GraphWrapper calls without changing decisions."""
from collections import Counter
import inspect
import json
import os
from pathlib import Path
import threading


class Recorder:
    def __init__(self, directory, marker):
        self.directory = Path(directory)
        self.marker = Path(marker)
        self.local = threading.local()
        self.rows = Counter()
        self.outside = Counter()
        self.calls = 0
        self.last_label = None
        self.sealed_labels = []
        self.errors = []

    def add(self, frame):
        key = json.dumps(frame, sort_keys=True, separators=(',', ':'))
        self.rows[key] += 1
        self.calls += 1

    def dump(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f'coverage_{os.getpid()}.json'
        data = dict(calls=self.calls, current_label=self.last_label,
                    sealed_labels=list(self.sealed_labels),
                    rows=[dict(frame=json.loads(k), count=v) for k, v in self.rows.items()],
                    outside_execute=dict(self.outside), errors=list(self.errors),
                    scope='Host API decisions and actual GraphWrapper call branches; no GPU event/kernel timing',
                    changed_dispatch=False, synchronized_GPU=False,
                    overhead_measured=False, complete=False)
        tmp = destination.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2) + '\n')
        tmp.replace(destination)

    def label(self):
        # Deliberate diagnostic I/O; measured throughput is not an optimization score.
        value = self.marker.read_text().strip() if self.marker.exists() else 'not_armed'
        assert value and len(value) <= 80
        if value != self.last_label:
            assert value not in self.sealed_labels, 'Diagnostic label reused'
            if self.last_label is not None:
                self.sealed_labels.append(self.last_label)
            self.dump()  # Snapshot the complete preceding label before the next request.
            self.last_label = value
        return value


def install(runner_class, graph_class, get_context, directory, marker):
    recorder = Recorder(directory, marker)
    original_execute = runner_class.execute_model
    original_dispatch = runner_class._determine_batch_execution_and_padding
    original_graph = graph_class.__call__
    dispatch_signature = inspect.signature(original_dispatch)

    def execute(owner, scheduler_output, *args, **kwargs):
        assert getattr(recorder.local, 'frame', None) is None
        scheduled = scheduler_output.num_scheduled_tokens
        frame = dict(label=recorder.label(), scheduled_requests=len(scheduled),
                     scheduled_tokens=sum(scheduled.values()),
                     scheduled_query_histogram=dict(Counter(int(v) for v in scheduled.values())),
                     draft_requests=len(scheduler_output.scheduled_spec_decode_tokens),
                     draft_tokens=sum(len(v) for v in scheduler_output.scheduled_spec_decode_tokens.values()),
                     dispatches=[], graph_calls=[])
        recorder.local.frame = frame
        try:
            return original_execute(owner, scheduler_output, *args, **kwargs)
        except BaseException as error:
            frame['error'] = repr(error)
            recorder.errors.append(repr(error))
            raise
        finally:
            recorder.local.frame = None
            recorder.add(frame)
            if recorder.calls % 64 == 0 or 'error' in frame:
                recorder.dump()

    def dispatch(owner, *args, **kwargs):
        result = original_dispatch(owner, *args, **kwargs)
        frame = getattr(recorder.local, 'frame', None)
        if frame is None:
            recorder.outside['dispatch:' + result[0].name] += 1
            return result
        bound = dispatch_signature.bind(owner, *args, **kwargs).arguments
        count = int(bound['num_reqs'])
        batch = owner.input_batch
        computed = batch.num_computed_tokens_cpu[:count]
        prompts = batch.num_prompt_tokens[:count]
        unfinished = sum(int(a < b) for a, b in zip(computed, prompts))
        desc = result[1]
        frame['dispatches'].append(dict(
            mode=result[0].name, tokens=int(bound['num_tokens']), requests=count,
            max_query=int(bound['max_num_scheduled_tokens']),
            query_histogram=dict(Counter(int(v) for v in bound['num_scheduled_tokens_np'])),
            unfinished_prompt_requests=unfinished,
            stage='prefill_or_mixed' if unfinished else 'decode_or_verify',
            uniform_query_length=int(owner.uniform_decode_query_len),
            force_uniform=bound.get('force_uniform_decode'), force_eager=bound.get('force_eager', False),
            padded_tokens=int(desc.num_tokens), padded_requests=desc.num_reqs,
            descriptor_uniform=desc.uniform))
        return result

    def graph(owner, *args, **kwargs):
        frame = getattr(recorder.local, 'frame', None)
        context = get_context()
        if context is None:
            branch = 'no_context'
            mode = None
        else:
            mode = context.cudagraph_runtime_mode.name
            if mode == 'NONE' or context.cudagraph_runtime_mode != owner.runtime_mode:
                branch = 'eager_passthrough'
            else:
                entry = owner.concrete_graph_entries.get(context.batch_descriptor)
                branch = 'capture' if entry is None or entry.graph is None else 'existing_graph_replay'
        result = original_graph(owner, *args, **kwargs)
        # Successful original __call__ + a pre-existing entry in the matching mode
        # reaches entry.graph.replay() in the pinned FL source. No proxy is installed.
        if frame is None:
            recorder.outside['graph:' + branch] += 1
        else:
            frame['graph_calls'].append(dict(branch=branch, context_mode=mode,
                                             wrapper_mode=owner.runtime_mode.name))
        return result

    runner_class.execute_model = execute
    runner_class._determine_batch_execution_and_padding = dispatch
    graph_class.__call__ = graph
    return recorder
