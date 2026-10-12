# SPDX-License-Identifier: Apache-2.0
"""Gate independent Math and unchanged official4K on a clean Decode service gain."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--service-results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    base=Path(__file__).resolve().parents[1]
    deadline=datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    state={'phase':'waiting','runner_pid':os.getpid(),'jobs':[],'accuracy_started':False,'official_started':False}
    def remaining():return (deadline-datetime.now(timezone.utc)).total_seconds()
    def save():
        p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')
    def run(name,command):
        assert remaining()>0
        state.update(phase='running',current_job=name);save()
        with (root/(name+'_runner.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd='/workspace/vllm-plugin-FL')
        state['jobs'].append({'name':name,'exit_code':result.returncode});save()
        assert result.returncode==0,name+' failed'
    save()
    try:
        p=args.service_results/'result.json'
        while not p.exists():
            assert remaining()>0
            time.sleep(5)
        status=json.loads(p.read_text())
        if status['phase']!='complete':
            state.update(phase='skipped',reason='Service confirmation failed')
        else:
            comparison=json.loads((args.service_results/'comparison.json').read_text())
            if not comparison['promising'] or not comparison['measured_tuning_clean']:
                state.update(phase='skipped',reason='No clear clean measured improvement')
            elif remaining()<5400:
                state.update(phase='skipped',reason='Less than90min remains for independent quality and official4K')
            else:
                candidate=args.service_results/'candidate'
                candidate_state=json.loads((candidate/'result.json').read_text())
                hook=base/'tianshu_decode_launch_20261012/hooks/sitecustomize.py'
                assert hashlib.sha256(hook.read_bytes()).hexdigest()==candidate_state['decode_hook']['sha256']
                assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
                quality=root/'math';state['accuracy_started']=True;save()
                run('math',['python3',str(base/'tianshu_candidate_quality_20261012/run_math_eval.py'),
                    '--output',str(quality),'--candidate','decode-launch','--adaptive','1','--max-batched-tokens','2048'])
                run('math_audit',['python3',str(base/'tianshu_candidate_quality_20261012/audit_math_eval.py'),
                    '--output',str(quality)])
                audit=json.loads((quality/'accuracy_audit.json').read_text())
                state.update(math_correct=audit['correct'],math_total=105,quality_passed=audit['official_requirement_passed'])
                if audit['official_requirement_passed'] and remaining()>2400:
                    state['official_started']=True;save();official=root/'official'
                    run('official',['python3',str(base/'tianshu_native_prefill_20261012/run_official_4k.py'),
                        '--output',str(official),'--control-trial',comparison['control_source'],
                        '--candidate-trial',str(candidate),'--max-batched-tokens','2048','--decode-launch','1',
                        '--math-results',str(quality),'--minimum-gain-percent','3'])
                    run('official_audit',['python3',str(base/'tianshu_native_prefill_20261012/audit_official.py'),
                        '--output',str(official),'--math-results',str(quality),
                        '--previous-best-tok-s','2712.55','--previous-best-ttft-ms','4720.38'])
                state['phase']='complete'
    except Exception as error:state.update(phase='failed',error=str(error),error_type=type(error).__name__)
    save();(root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['phase'] in ('complete','skipped') else 1


if __name__=='__main__':raise SystemExit(main())
