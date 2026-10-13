# SPDX-License-Identifier: Apache-2.0
"""Original official benchmark with source-matched address32 quality and routing."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--service-root',type=Path,required=True)
    parser.add_argument('--math-results',type=Path,required=True)
    parser.add_argument('--input-tokens',type=int,choices=[4096,16384],required=True)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    service=json.loads((args.service_root/'candidate/result.json').read_text())
    compared=json.loads((args.service_root/'comparison.json').read_text())
    assert compared['complete'] and compared['gain_percent']>=2
    audit=json.loads((args.math_results/'accuracy_audit.json').read_text())
    assert audit['fresh_math_for_address32'] and audit['official_requirement_passed']
    base=Path(__file__).resolve().parents[1]
    name='run_official_4k.py' if args.input_tokens==4096 else 'run_official_16k.py'
    trial=base/'tianshu_native_prefill_20261012'/name
    hook=Path(__file__).resolve().parent/'hooks/sitecustomize.py'
    sources={'address32_sitecustomize.py':hook,'address32.py':hook.parent.parent/'address32.py'}
    snapshot=root/'address_source_snapshot';snapshot.mkdir();provenance={}
    for label,p in sources.items():
        raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        assert sha==audit['address32_sources'][label]['sha256']==service['address32_sources'][label]['sha256']
        (snapshot/label).write_bytes(raw);provenance[label]=dict(path=str(p),sha256=sha)
    for p in [Path(__file__),trial]:(snapshot/p.name).write_bytes(p.read_bytes())
    original=subprocess.Popen
    def popen(command,*positional,**kwargs):
        if isinstance(command,list) and command[:2]==['vllm','serve']:
            env=dict(kwargs['env']);env.update(PYTHONPATH=str(hook.parent),TIANSHU_ADDRESS32_PROBE='1',
                                             TIANSHU_ADDRESS32_RECORD_DIR=str(root/'address32_records'))
            kwargs['env']=env
        return original(command,*positional,**kwargs)
    spec=importlib.util.spec_from_file_location('_tianshu_address_official_trial',trial)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    subprocess.Popen=popen
    control=Path(compared['control_root']) if compared.get('control_reused') else args.service_root/'control'
    sys.argv=[str(trial),'--output',str(root),'--control-trial',str(control),'--candidate-trial',str(args.service_root/'candidate'),
              '--math-results',str(args.math_results),'--max-batched-tokens','2048','--decode-launch','1',
              '--buckets','1','--gather','1','--bucket-trace','2']
    try:code=module.main()
    finally:subprocess.Popen=original
    state=json.loads((root/'result.json').read_text())
    records=[json.loads(f.read_text()) for f in (root/'address32_records').glob('route_*.json')]
    state.update(address32_sources=provenance,address32_routing_records=records,
                 address32_environment=dict(PYTHONPATH=str(hook.parent),TIANSHU_ADDRESS32_PROBE='1',TIANSHU_ADDRESS32_RECORD_DIR=str(root/'address32_records')),
                 matched_service_root=str(args.service_root),matched_math_root=str(args.math_results))
    (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    if code==0:
        assert records and all(4<=r['batch']<=48 and max(r['bounds'])<2**31 and r['kv_quant_mode']==0 for r in records)
        assert {r['generated_sha256'] for r in records}=={r['generated_sha256'] for r in audit['address32_routing_records']}
        script='audit_official.py' if args.input_tokens==4096 else 'audit_official_16k.py'
        previous=('3222.52','4510.23') if args.input_tokens==4096 else ('2440.62','200982.18')
        subprocess.run(['python3',str(base/'tianshu_native_prefill_20261012'/script),'--output',str(root),
                        '--math-results',str(args.math_results),'--previous-best-tok-s',previous[0],
                        '--previous-best-ttft-ms',previous[1]],check=True,capture_output=True)
        subprocess.run(['python3',str(base/'archive_completed_pair.py'),'--single-probe','--source',str(root),
                        '--label',f'attention_address32_official{args.input_tokens}_20261013'],check=True)
    return code


if __name__=='__main__':raise SystemExit(main())
