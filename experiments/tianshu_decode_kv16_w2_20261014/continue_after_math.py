# SPDX-License-Identifier: Apache-2.0
"""Lease-bounded serial handoff from fresh Math105 to original official4K."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--math-root', type=Path, required=True)
    parser.add_argument('--confirm-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists() and not args.official_root.exists()
    root.mkdir(parents=True)
    base = Path(__file__).resolve().parent
    paths = {base/'run_official_4k.py':'743dc3e4ec4ba4040140dfa4296c6eb1a35df45935fe2e5e53cd19d1885eb34f',
             base/'audit_official.py':'7b75e204e9e70f27d38b8339c0c42ff5658643549f352884e9442ea78539e093'}
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert frozen == {str(p):s for p,s in paths.items()}
    paths[Path(__file__).resolve()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    snapshot = root/'source_snapshot';snapshot.mkdir()
    for p in paths:(snapshot/p.name).write_bytes(p.read_bytes())
    deadline = datetime.fromisoformat(json.loads(Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json').read_text())['stop_at_utc'])
    remaining = int((deadline-datetime.now(timezone.utc)).total_seconds())
    assert remaining > 3600
    state = dict(phase='waiting_for_math',runner_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),stop_at_utc=deadline.isoformat(),official_benchmark_started=False,adopted=False,frozen_sources={str(p):s for p,s in paths.items()})
    child = None
    class StopRequested(BaseException):pass
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise StopRequested(str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(signum,stop)
    signal.alarm(remaining);save()
    try:
        while True:
            assert datetime.now(timezone.utc)<deadline
            math = json.loads((args.math_root/'state.json').read_text())
            proc = Path('/proc')/str(math['runner_pid'])/'stat'
            proc_state = proc.read_text().split()[2] if proc.exists() else 'missing'
            state.update(math_phase=math['phase'],math_runner_pid=math['runner_pid'],math_proc_state=proc_state)
            save()
            if math['phase'] in ('failed','stopped'):raise RuntimeError('Math terminated: '+str(math.get('error')))
            archive = Path('/tmp/tianshu_decode_kv16_w1_math_20261014_manifest.json')
            if proc_state in ('Z','missing'):
                if not archive.exists():raise RuntimeError('Math runner ended without completed accuracy archive')
                break
            time.sleep(10)
        audit = json.loads((args.math_root/'accuracy_audit.json').read_text())
        assert audit['complete'] and audit['total']==105 and audit['fresh_math_for_decode_kv16_w1']
        assert not audit['reused_original_accuracy'] and audit['execution_summary']['errored']==0
        if not audit['official_requirement_passed'] or audit['correct']/105<=.95:
            state.update(phase='quality_rejected',math_correct=audit['correct'],math_total=105)
            save();return 0
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in audit['kv16_sources'].items())
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==sha for p,sha in paths.items())
        assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
        # Re-audit actual reviews/predictions and frozen computation before starting a new GPU job.
        for name,meta in audit['raw_files'].items():
            p=args.math_root/name
            assert p.stat().st_size==meta['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
        review_paths=list(args.math_root.glob('evalscope/*/reviews/minicpm/math_500_Level 3.jsonl'))
        prediction_paths=list(args.math_root.glob('evalscope/*/predictions/minicpm/math_500_Level 3.jsonl'))
        assert len(review_paths)==len(prediction_paths)==1
        reviews=[json.loads(x) for x in review_paths[0].read_text().splitlines()]
        predictions=[json.loads(x) for x in prediction_paths[0].read_text().splitlines()]
        assert len(reviews)==len(predictions)==105
        assert len({x['index'] for x in reviews})==len({x['index'] for x in predictions})==105
        assert {x['index'] for x in reviews}=={x['index'] for x in predictions}
        assert all(x['model_output']['error'] is None for x in predictions)
        assert all(x['sample_score']['score']['status']=='success' for x in reviews)
        scores=[x['sample_score']['score']['value']['accuracy'] for x in reviews]
        assert all(x in (0,1) for x in scores) and sum(scores)==audit['correct']
        (root/'math_readonly_reaudit.json').write_text(json.dumps(dict(complete=True,raw_files_sha256_verified=True,reviews=105,predictions=105,correct=sum(scores),quality_passed=True),indent=2)+'\n')
        command=['python3','-u',str(base/'run_official_4k.py'),'--output',str(args.official_root),
                 '--control-trial',str(args.confirm_root/'control'),'--candidate-trial',str(args.confirm_root/'candidate'),
                 '--math-results',str(args.math_root),'--max-batched-tokens','2048','--decode-launch','1',
                 '--buckets','1','--gather','1','--bucket-trace','2']
        with (root/'official_runner.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/workspace')
            state.update(phase='running_official',child_pid=child.pid,official_benchmark_started=True,math_correct=audit['correct'],math_total=105,command=command);save()
            code=child.wait()
        state['official_runner_exit_code']=code
        result=json.loads((args.official_root/'result.json').read_text())
        assert code==0 and result['phase']=='complete'
        state.update(phase='complete',official_root=str(args.official_root),local_archive_download_audit_still_required=True)
    except StopRequested as error:state.update(phase='stopped',error=str(error))
    except Exception as error:state.update(phase='failed',error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase'] in ('complete','quality_rejected') else 1


if __name__=='__main__':
    raise SystemExit(main())
