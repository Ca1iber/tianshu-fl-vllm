# SPDX-License-Identifier: Apache-2.0
"""Own one short GPU child; preserve source hashes and terminal evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-at-utc', required=True)
    args = parser.parse_args(); root = args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists()
    remaining = int((datetime.fromisoformat(args.stop_at_utc)-datetime.now(timezone.utc)).total_seconds())
    assert remaining > 1200
    root.mkdir(parents=True)
    base = Path(__file__).resolve().parent
    sources = [Path(__file__), base/'log2_decode.py', base/'probe_log2.py',
        base.parent/'tianshu_scalar_kv_page_20261016/probe_page.py',
        base.parent/'tianshu_native_prefill_20261012/zero_initialization.py',
        base.parent/'tianshu_decode_cache_policy_20261014/experimental_cache.py',
        base.parent/'tianshu_decode_low_warps_20261014/launch_router.py',
        Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py'),
        Path('/usr/local/lib/python3.12/site-packages/triton/backends/iluvatar/compiler.py'),
        Path('/usr/local/lib/python3.12/site-packages/triton/compiler/compiler.py'),
        Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_attention_helpers.py')]
    frozen = {str(p): digest(p) for p in sources}
    for p in sources:
        out = root/'source_snapshot'/p.relative_to('/');out.parent.mkdir(parents=True, exist_ok=True);out.write_bytes(p.read_bytes())
    child = None
    state = dict(phase='preparing', complete=False, official_benchmark=False, adopted=False,
                 runner_pid=os.getpid(), frozen_sources=frozen, stop_at_utc=args.stop_at_utc,
                 packages={name: importlib.metadata.version(name) for name in ('torch','vllm','flag-gems')})
    def save():
        (root/'state.json').write_text(json.dumps(state, indent=2)+'\n')
    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise KeyboardInterrupt(str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):signal.signal(sig, stop)
    signal.alarm(min(remaining, 1500)); save()
    try:
        hardware = subprocess.check_output(['ixsmi'], text=True)
        (root/'hardware_before.txt').write_text(hardware)
        assert 'No running processes found' in hardware
        command = [sys.executable, '-u', str(base/'probe_log2.py'), '--output', str(root/'micro'),
                   '--zero-init', '--stop-at-utc', args.stop_at_utc]
        env = dict(os.environ); env.pop('PYTHONPATH', None)
        with (root/'probe.log').open('w') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(phase='micro', child_pid=child.pid, command=command); save()
            code = child.wait(timeout=min(remaining-30, 1200))
        state['child_exit_code'] = code
        assert code == 0
        result = json.loads((root/'micro/result.json').read_text())
        assert result['complete'] and result['numerical_checks_passed'] and len(result['cases']) == 21
        for p, h in frozen.items():assert digest(Path(p)) == h, p
        state.update(phase='complete', complete=True, candidate_cases=42)
    except BaseException as error:
        state.update(phase='failed', error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid, signal.SIGKILL);child.wait(timeout=5)
        signal.alarm(0)
        (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'], text=True)); save()
        (root/'result.json').write_text(json.dumps(state, indent=2)+'\n')
    if state['complete']:
        assert 'No running processes found' in (root/'hardware_after.txt').read_text()
        subprocess.run([sys.executable, str(base.parent/'archive_completed_pair.py'), '--single-probe',
                        '--source', str(root), '--label', 'log2_decode_20261016'], check=True)
    return 0 if state['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
