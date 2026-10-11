# SPDX-License-Identifier: Apache-2.0
"""Serialized short 4K Prefill budget sweep, no official score."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--adaptive',choices=['0','1'],default='0');args=parser.parse_args()
 root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
 trial=Path(__file__).with_name('run_budget_trial.py')
 for budget in [2048,4096,8192]:
  out=root/f'budget_{budget}';out.mkdir(exist_ok=True)
  with (out/'runner.log').open('w') as log:
   result=subprocess.run([sys.executable,'-u',str(trial),'--output',str(out),'--buckets','0','--gather','0','--hybrid','1','--adaptive',args.adaptive,'--max-batched-tokens',str(budget),'--output-tokens','128','--concurrency','16','--requests','16','--rounds','5'],stdout=log,stderr=subprocess.STDOUT)
  data=json.loads((out/'result.json').read_text());print(json.dumps({'budget':budget,'phase':data['phase'],'returncode':result.returncode,'stability':data.get('measured_round_stability')},ensure_ascii=False),flush=True)
  if result.returncode!=0 or data['phase']!='complete':raise RuntimeError('Budget '+str(budget)+' failed; preserve logs')
  if 'No running processes found' not in subprocess.check_output(['ixsmi'],text=True):raise RuntimeError('GPU not idle after owned service cleanup')


if __name__=='__main__':main()
