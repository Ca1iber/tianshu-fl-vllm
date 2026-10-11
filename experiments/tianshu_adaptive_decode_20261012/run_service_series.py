# SPDX-License-Identifier: Apache-2.0
"""Serial Hybrid S16 versus adaptive Decode trials; no official score."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
 script=Path(__file__).with_name('run_service_comparison.py')
 for flag,name in [('0','service_control'),('1','service_adaptive')]:
  out=root/name;out.mkdir(exist_ok=True)
  with (out/'runner.log').open('w') as log:
   completed=subprocess.run([sys.executable,'-u',str(script),'--output',str(out),'--buckets','0','--gather','0','--hybrid','1','--adaptive',flag,'--output-tokens','1024','--concurrency','64','--requests','64','--rounds','3','--single-probe-output','4096'],stdout=log,stderr=subprocess.STDOUT)
  data=json.loads((out/'result.json').read_text());print(json.dumps({'name':name,'phase':data['phase'],'exit':completed.returncode,'measured':data.get('measured_round_stability')},ensure_ascii=False),flush=True)
  if completed.returncode!=0 or data['phase']!='complete':raise RuntimeError(name+' failed')
  if 'No running processes found' not in subprocess.check_output(['ixsmi'],text=True):raise RuntimeError('GPU not idle after owned cleanup')


if __name__=='__main__':main()
