# SPDX-License-Identifier: Apache-2.0
"""Current best operators plus bounded prefill cadence, observed in both arms."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[1]
MODE = os.environ['TIANSHU_PREFILL_CADENCE_MODE']
assert MODE in ('control', 'candidate')
assert os.environ['TIANSHU_TOPP_PRECISION_SERVICE_MODE'] == 'candidate'
assert os.environ['TIANSHU_PREFILL_GEMM_TILES_PROBE'] == '1'
runpy.run_path(str(BASE.parent / 'tianshu_gemm_topp_geo_20261016/hooks/sitecustomize.py'))
SOURCE_SHA = 'df78076164fe02a63876d7690cefac810f1ba4d8fb58afc15e786c1608064090'


class Loader(importlib.abc.Loader):
    def __init__(self, original): self.original = original
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        source = Path(module.__file__)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
        original = module.Scheduler.schedule
        signature = inspect.signature(original)
        assert list(signature.parameters) == ['self', 'throttle_prefills']
        assert signature.parameters['throttle_prefills'].default is False
        spec = importlib.util.spec_from_file_location('_tianshu_prefill_cadence', BASE / 'policy.py')
        policy = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = policy
        spec.loader.exec_module(policy)
        directory = Path(os.environ['TIANSHU_PREFILL_CADENCE_RECORD_DIR'])
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(mode=MODE, pid=os.getpid(), source=str(source), source_sha256=SOURCE_SHA,
                      policy_sha256=hashlib.sha256((BASE / 'policy.py').read_bytes()).hexdigest(),
                      calls=0, supported=0, pressure=0, deferred_steps=0, held_requests=0,
                      mrv1_metadata_steps=0, history_payload_steps=0,
                      scheduled_held_requests=0, empty_steps=0, step_shapes={}, original_args_forwarded=True,
                      max_consecutive_deferral=2, eligibility_restored_after_original=True)
        journal = (directory / f'steps_{os.getpid()}.jsonl').open('a', buffering=1)
        def save():
            p = directory / f'runtime_{os.getpid()}.json'
            tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(record, indent=2) + '\n'); tmp.replace(p)
        def observe(owner, output, decision):
            if record['calls']==0:
                hf=owner.vllm_config.model_config.hf_config
                record['configuration']=dict(configured_use_v2=owner.use_v2_model_runner,
                    worker_cls=owner.parallel_config.worker_cls,block_size=owner.block_size,
                    hf_geometry=[hf.hidden_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads],
                    async_scheduling=owner.scheduler_config.async_scheduling,
                    max_model_len=owner.max_model_len,max_num_scheduled_tokens=owner.max_num_scheduled_tokens)
            record['calls'] += 1
            record['supported'] += int(decision['supported'])
            record['pressure'] += int(decision['pressure'])
            record['deferred_steps'] += int(decision['defer'])
            record['held_requests'] += decision['held']
            record['scheduled_held_requests'] += decision['scheduled_held_requests']
            record['mrv1_metadata_steps'] += int(decision['used_mrv1_metadata'])
            history=len(output.scheduled_cached_reqs.all_token_ids)
            record['history_payload_steps'] += int(history>0)
            counts = list(output.num_scheduled_tokens.values())
            key = ','.join(map(str, (len(counts), sum(counts), max(counts, default=0))))
            record['step_shapes'][key] = record['step_shapes'].get(key, 0) + 1
            record['empty_steps'] += int(not counts)
            journal.write(json.dumps(dict(**decision, sequences=len(counts),
                                          tokens=sum(counts), max_query=max(counts, default=0),
                                          history_payload_entries=history)) + '\n')
            if record['calls'] == 1 or record['calls'] % 128 == 0:
                save()
        module.Scheduler.schedule = policy.wrap(original, enabled=MODE == 'candidate', observe=observe)
        save()


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm.v1.core.sched.scheduler': return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


sys.meta_path.insert(0, Finder())
