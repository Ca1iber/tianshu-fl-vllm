# SPDX-License-Identifier: Apache-2.0
"""Process-only replacement of the gather helper for an explicit experiment."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys


class GatherOverrideLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        super().exec_module(module)
        original = module.enabled
        recorded = set()

        def enabled(config, platform):
            decision = original(config, platform)
            directory = os.environ.get('TIANSHU_GATHER_PROBE_RECORD_DIR')
            if directory and decision not in recorded:
                recorded.add(decision)
                output = Path(directory)
                output.mkdir(parents=True, exist_ok=True)
                record = {'pid': os.getpid(), 'enabled': decision,
                          'budget': config.scheduler_config.max_num_batched_tokens,
                          'dtype': str(config.model_config.dtype),
                          'vendor': getattr(platform, 'vendor_name', None),
                          'source': module.__file__}
                (output / f'routing_{os.getpid()}_{decision}.json').write_text(
                    json.dumps(record, indent=2) + '\n')
            return decision

        module.enabled = enabled


class GatherOverrideFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm_fl.worker.iluvatar_logits_gather':
            return None
        source = Path(os.environ['TIANSHU_GATHER_PROBE_SOURCE']).resolve()
        if not source.is_file():
            raise RuntimeError('Missing explicit experimental gather source')
        directory = os.environ.get('TIANSHU_GATHER_PROBE_RECORD_DIR')
        if directory:
            output = Path(directory)
            output.mkdir(parents=True, exist_ok=True)
            record = {'pid': os.getpid(), 'module': fullname, 'source': str(source),
                      'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                      'gather_flag': os.environ.get('VLLM_FL_ILUVATAR_LOGITS_GATHER'),
                      'bucket_flag': os.environ.get('VLLM_FL_ILUVATAR_PREFILL_BUCKETS')}
            (output / f'override_{os.getpid()}.json').write_text(json.dumps(record, indent=2) + '\n')
        return importlib.util.spec_from_file_location(
            fullname, source, loader=GatherOverrideLoader(fullname, str(source)))


if os.environ.get('TIANSHU_GATHER_PROBE_SOURCE'):
    sys.meta_path.insert(0, GatherOverrideFinder())
