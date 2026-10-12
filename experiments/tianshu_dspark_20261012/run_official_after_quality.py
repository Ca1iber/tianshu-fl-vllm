# SPDX-License-Identifier: Apache-2.0
"""Start exactly one official4K only after a complete promising/quality audit."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--standard-results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();standard=args.standard_results.resolve();root=args.output.resolve()
    assert not root.exists();root.mkdir(parents=True)
    lease=Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    deadline=datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])
    state={'phase':'waiting','runner_pid':os.getpid(),'official_benchmark_started':False}
    def save():
        p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'controller_state.json')
    save()
    try:
        while True:
            assert datetime.now(timezone.utc)<deadline,'Reservation ended'
            p=standard/'state.json'
            if p.exists():
                status=json.loads(p.read_text())
                if status['phase'] not in ('waiting','running'):break
            time.sleep(5)
        quality=standard/'math/accuracy_audit.json'
        if status['phase']!='complete' or not quality.exists():
            state.update(phase='skipped',reason='Standard confirmation or independent quality not completed')
        elif not json.loads(quality.read_text())['official_requirement_passed']:
            state.update(phase='skipped',reason='Independent accuracy is not strictly greater than95%')
        else:
            comparison=json.loads((standard/'comparison.json').read_text())
            assert comparison['promising'] and comparison['measured_tuning_clean']
            assert (deadline-datetime.now(timezone.utc)).total_seconds()>3600,'Less than1h left for official4K'
            flags=status['candidate_flags']
            state.update(phase='running',official_benchmark_started=True,candidate_flags=flags);save()
            directory=Path(__file__).resolve().parent
            output=root/'official'
            command=['python3',str(directory/'run_smoke.py'),'--output',str(output),
                     '--official-4k','--quality-results',str(standard/'math'),
                     '--standard-results',str(standard)]+flags
            with (root/'official_runner.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd='/workspace/vllm-plugin-FL')
            state['official_exit_code']=result.returncode;assert result.returncode==0,'Official run failed'
            audit=['python3',str(directory/'audit_official.py'),'--output',str(output),
                   '--quality-results',str(standard/'math')]
            with (root/'audit.log').open('w') as log:
                result=subprocess.run(audit,stdout=log,stderr=subprocess.STDOUT,cwd='/workspace/vllm-plugin-FL')
            state['audit_exit_code']=result.returncode;assert result.returncode==0,'Official audit failed'
            state.update(phase='complete',comparison=json.loads((output/'comparison.json').read_text()))
    except Exception as error:
        state.update(phase='failed',error=str(error))
    save();(root/'controller_result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase'] in ('complete','skipped') else 1


if __name__=='__main__':main()
