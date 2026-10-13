# SPDX-License-Identifier: Apache-2.0
"""Fresh official Level3 evaluation with exactly the measured address32 hook."""
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
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    compared=json.loads((args.service_root/'comparison.json').read_text())
    assert compared['complete'] and compared['gain_percent']>0
    service=json.loads((args.service_root/'candidate/result.json').read_text())
    base=Path(__file__).resolve().parents[1]
    trial=base/'tianshu_candidate_quality_20261012/run_math_eval.py'
    hook=Path(__file__).resolve().parent/'hooks/sitecustomize.py'
    sources={'address32_sitecustomize.py':hook,'address32.py':hook.parent.parent/'address32.py'}
    snapshot=root/'address_source_snapshot';snapshot.mkdir()
    provenance={}
    for name,p in sources.items():
        raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        assert sha==service['address32_sources'][name]['sha256']
        (snapshot/name).write_bytes(raw);provenance[name]=dict(path=str(p),sha256=sha)
    for p in [Path(__file__),trial]:(snapshot/p.name).write_bytes(p.read_bytes())
    original=subprocess.Popen
    def popen(command,*positional,**kwargs):
        if isinstance(command,list) and command[:2]==['vllm','serve']:
            env=dict(kwargs['env']);env.update(PYTHONPATH=str(hook.parent),TIANSHU_ADDRESS32_PROBE='1',
                                             TIANSHU_ADDRESS32_RECORD_DIR=str(root/'address32_records'))
            kwargs['env']=env
        return original(command,*positional,**kwargs)
    spec=importlib.util.spec_from_file_location('_tianshu_address_math_trial',trial)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    subprocess.Popen=popen
    sys.argv=[str(trial),'--output',str(root),'--candidate','decode-launch','--buckets','1','--gather','1',
              '--bucket-trace','2','--max-batched-tokens','2048']
    try:code=module.main()
    finally:subprocess.Popen=original
    state=json.loads((root/'result.json').read_text())
    records=[json.loads(f.read_text()) for f in (root/'address32_records').glob('route_*.json')]
    state.update(address32_sources=provenance,address32_routing_records=records,
                 address32_environment=dict(PYTHONPATH=str(hook.parent),TIANSHU_ADDRESS32_PROBE='1',TIANSHU_ADDRESS32_RECORD_DIR=str(root/'address32_records')),
                 matched_service_root=str(args.service_root),fresh_math_for_address32=True)
    (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    if code==0:
        assert records and all(4<=r['batch']<=48 and max(r['bounds'])<2**31 and r['kv_quant_mode']==0 for r in records)
        assert {r['generated_sha256'] for r in records}=={r['generated_sha256'] for r in service['address32_routing_records']}
        subprocess.run(['python3',str(base/'tianshu_candidate_quality_20261012/audit_math_eval.py'),
                        '--output',str(root)],check=True,capture_output=True)
        audit=json.loads((root/'accuracy_audit.json').read_text())
        audit.update(address32_sources=provenance,address32_routing_records=records,
                     address32_environment=state['address32_environment'],
                     matched_service_root=str(args.service_root),fresh_math_for_address32=True)
        (root/'accuracy_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        subprocess.run(['python3',str(base/'archive_completed_pair.py'),'--single-probe','--source',str(root),
                        '--completion-file','accuracy_audit.json','--label','attention_address32_math_20261013'],check=True)
    return code


if __name__=='__main__':raise SystemExit(main())
