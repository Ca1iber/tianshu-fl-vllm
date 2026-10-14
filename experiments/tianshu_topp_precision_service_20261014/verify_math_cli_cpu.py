# SPDX-License-Identifier: Apache-2.0
"""Exercise real Math entrypoint guards and argv up to intercepted server Popen.

Explicit synthetic five-round fixture tests orchestration only. It cannot establish
service gain, accuracy, server startup, backend support, or any GPU result.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    runner = base/'run_math_eval.py'
    spec = importlib.util.spec_from_file_location('_budget_math_cpu', runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    class CPUStop(BaseException):
        pass

    def popen(command, **kwargs):
        assert command[:2] == ['vllm', 'serve']
        assert command[command.index('--max-num-batched-tokens')+1] == '4096'
        assert command[command.index('--long-prefill-token-threshold')+1] == '0'
        assert command[command.index('--max-num-seqs')+1] == '64'
        env = kwargs['env']
        for name in ('TIANSHU_SQL_REFLECTION_PROBE','TIANSHU_EXTENDED_BUCKET_PROBE',
                     'TIANSHU_SEGMENT_POLICY_PROBE','TIANSHU_DECODE_COHERENT_PROBE',
                     'TIANSHU_TOPP_LAUNCH_PROBE','VLLM_FL_ILUVATAR_PREFILL_BUCKETS',
                     'VLLM_FL_ILUVATAR_LOGITS_GATHER'):
            assert env[name] == '1', name
        assert env['TIANSHU_KV16_SERVICE_MODE'] == 'candidate'
        assert env['TIANSHU_CACHE_SERVICE_MODE'] == 'candidate'
        assert env['TIANSHU_TOPP_PRECISION_SERVICE_MODE'] == 'candidate' and env['TIANSHU_MAX_SEQS_MODE'] == 'candidate'
        assert env['PYTHONPATH'] == str(base/'hooks')
        captured.update(command=command, environment={k:v for k,v in env.items()
                        if k.startswith(('TIANSHU_','VLLM_FL_','PYTHONPATH'))})
        raise CPUStop('CPU-only interception before any server process')

    def check_output(command, **kwargs):
        assert command == ['ixsmi'], command
        return 'No running processes found (CPU fixture; no hardware query)'

    with tempfile.TemporaryDirectory(prefix='budget_math_cli_cpu_') as temp:
        temp = Path(temp)
        service = temp/'SYNTHETIC_SERVICE_FIXTURE';service.mkdir()
        (service/'candidate').mkdir()
        comparison = json.loads((args.screen_root/'comparison.json').read_text())
        trial = json.loads((args.screen_root/'candidate/result.json').read_text())
        # Only exercise real preflight branches; no synthetic observations are
        # saved to performance roots or consumed by a live GPU continuation.
        trial['short_4k_rounds'] += copy.deepcopy(trial['short_4k_rounds'][-2:])
        comparison['excluded_rounds'] = [1,2]
        comparison['control_rounds'] = [100.] * 5
        comparison['candidate_rounds'] = [103.] * 5
        comparison['gain_percent'] = 3.
        (service/'comparison.json').write_text(json.dumps(comparison))
        (service/'candidate/result.json').write_text(json.dumps(trial))
        output = temp/'output'
        argv = [str(runner),'--output',str(output),'--service-root',str(service),
                '--candidate','decode-coherent','--max-batched-tokens','4096',
                '--buckets','1','--gather','1','--bucket-trace','2']
        with patch.object(sys,'argv',argv), patch.object(module.subprocess,'Popen',popen), \
             patch.object(module.subprocess,'check_output',check_output), \
             patch.object(module.socket,'create_connection',side_effect=OSError('CPU mock')), \
             patch.object(module.threading.Thread,'start',return_value=None):
            assert module.main() == 1
        state = json.loads((output/'result.json').read_text())
        assert state['phase'] == 'failed' and state['error'].startswith('CPU-only interception')
        assert len(state['segment_sources']) == 18
        assert captured['command'] == trial['command']
        for path,digest in state['segment_sources'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    args.output.parent.mkdir(parents=True,exist_ok=True)
    data = dict(complete=True,cpu_only=True,synthetic_fixture_not_performance=True,
                real_math_entrypoint_guards_exercised=True,
                intercepted_before_server_process=True,eighteen_sources_verified=True,
                GPU_quality_not_run=True,**captured,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (runner,base/'audit_math_eval.py',base/'audit_profile.py',Path(__file__))})
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k not in ('environment','command')}))


if __name__ == '__main__':
    main()
