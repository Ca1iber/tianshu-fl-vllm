# SPDX-License-Identifier: Apache-2.0
"""Own the component GPU process and archive only after its context exits."""
import argparse,json,os,signal,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);a=p.parse_args()
root=a.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
remaining=int((datetime.fromisoformat(a.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>1200
child=None;s=dict(phase='preparing',complete=False,runner_pid=os.getpid(),official_benchmark=False,adopted=False)
def save():(root/'state.json').write_text(json.dumps(s,indent=2)+'\n')
def stop(sig,frame):raise TimeoutError(sig)
for sig in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
signal.alarm(min(remaining,2100));save()
try:
 assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
 command=[sys.executable,'-u',str(Path(__file__).with_name('probe.py')),'--output',str(root/'micro'),'--stop-at-utc',a.stop_at_utc]
 with (root/'probe.log').open('w') as log:
  child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  s.update(phase='micro',child_pid=child.pid);save();assert child.wait(timeout=min(remaining-20,1900))==0
 d=json.loads((root/'micro/result.json').read_text());assert d['complete'] and d['phase']=='complete';s.update(phase='complete',complete=True)
except BaseException as error:s.update(phase='failed',error=repr(error))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
 signal.alarm(0);save();(root/'result.json').write_text(json.dumps(s,indent=2)+'\n')
 (root/'hardware_after.txt').write_text(subprocess.check_output(['ixsmi'],text=True))
if s['complete']:
 subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'archive_completed_pair.py'),'--single-probe','--source',str(root),'--label','exponential_key_warm_component_20261019'],check=True)
raise SystemExit(0 if s['complete'] else 1)
