# SPDX-License-Identifier: Apache-2.0
"""Two serialized short 4K service trials; no official score."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--requests', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=5)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).with_name('run_service_comparison.py')
    for flag,name in (('0','service_control'),('1','service_hybrid')):
        out=root/name;out.mkdir(exist_ok=True)
        with (out/'runner.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',str(script),'--output',str(out),
                '--buckets','0','--gather','0','--hybrid',flag,
                '--output-tokens',str(args.output_tokens),'--concurrency',str(args.concurrency),
                '--requests',str(args.requests),'--rounds',str(args.rounds)],stdout=log,stderr=subprocess.STDOUT)
        data=json.loads((out/'result.json').read_text())
        print(json.dumps({'name':name,'phase':data['phase'],'exit':result.returncode,
            'measured_rounds':data.get('measured_round_stability')},ensure_ascii=False),flush=True)
        if result.returncode!=0 or data['phase']!='complete':raise RuntimeError(name+' failed')
        if 'No running processes found' not in subprocess.check_output(['ixsmi'],text=True):
            raise RuntimeError('GPU not idle after owned server cleanup')
        if flag=='1':
            path=out/'routing_stats.json'
            if path.exists():
                counts=json.loads(path.read_text())['python_callbacks']
                if counts['prefill']+counts['mixed']==0:raise RuntimeError('Hybrid never selected')
                print('HYBRID_CALLBACKS',json.dumps(counts),flush=True)


if __name__=='__main__':main()
