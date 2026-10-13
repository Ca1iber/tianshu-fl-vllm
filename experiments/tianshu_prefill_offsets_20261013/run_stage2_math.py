# SPDX-License-Identifier: Apache-2.0
"""Fresh source-matched Math105 for Prefill stage2 plus guarded Decode32."""
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
    comparison=json.loads((args.service_root/'comparison.json').read_text());assert comparison['complete'] and comparison['gain_percent']>0
    service=json.loads((args.service_root/'candidate/result.json').read_text());assert service['prefill_stages']==2
    base=Path(__file__).resolve().parents[1];trial=base/'tianshu_candidate_quality_20261012/run_math_eval.py'
    hook=Path(__file__).resolve().parent/'hooks_stage2/sitecustomize.py'
    sources={'stage2_sitecustomize.py':hook,
             'address32_sitecustomize.py':base/'tianshu_attention_offsets_20261013/hooks/sitecustomize.py',
             'address32.py':base/'tianshu_attention_offsets_20261013/address32.py'}
    snapshot=root/'stage2_source_snapshot';snapshot.mkdir();provenance={}
    for name,path in sources.items():
        raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        expected=service['prefill_stage2_hook']['sha256'] if name=='stage2_sitecustomize.py' else service['address32_sources'][name]['sha256']
        assert sha==expected;(snapshot/name).write_bytes(raw);provenance[name]=dict(path=str(path),sha256=sha)
    for p in [Path(__file__),trial]:(snapshot/p.name).write_bytes(p.read_bytes())
    original=subprocess.Popen
    actual_env=dict(PYTHONPATH=str(hook.parent),TIANSHU_ADDRESS32_PROBE='1',
                    TIANSHU_ADDRESS32_RECORD_DIR=str(root/'address32_records'),
                    TIANSHU_PREFILL_STAGE2_PROBE='1',TIANSHU_PREFILL_STAGE2_RECORD_DIR=str(root/'prefill_stage2_records'))
    def popen(command,*positional,**kwargs):
        if isinstance(command,list) and command[:2]==['vllm','serve']:
            env=dict(kwargs['env']);env.update(actual_env);kwargs['env']=env
        return original(command,*positional,**kwargs)
    spec=importlib.util.spec_from_file_location('_tianshu_stage2_math_trial',trial)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);subprocess.Popen=popen
    sys.argv=[str(trial),'--output',str(root),'--candidate','decode-launch','--buckets','1','--gather','1','--bucket-trace','2','--max-batched-tokens','2048']
    try:code=module.main()
    finally:subprocess.Popen=original
    state=json.loads((root/'result.json').read_text())
    records=[json.loads(p.read_text()) for p in (root/'prefill_stage2_records').glob('configured_*.json')]
    addr=[json.loads(p.read_text()) for p in (root/'address32_records').glob('route_*.json')]
    state.update(prefill_stages=2,stage2_sources=provenance,stage2_environment=actual_env,
                 stage2_configuration_records=records,address32_routing_records=addr,
                 fresh_math_for_prefill_stage2=True,matched_service_root=str(args.service_root))
    (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    if code==0:
        assert records and addr and all(r['configuration']['num_stages']==2 for r in records)
        assert all(r['configuration']==dict(BLOCK_M=256,TILE_SIZE=128,num_warps=16,num_stages=2) for r in state['native_prefill_routing_records'])
        assert {r['generated_sha256'] for r in addr}=={r['generated_sha256'] for r in service['address32_routing_records']}
        subprocess.run(['python3',str(base/'tianshu_candidate_quality_20261012/audit_math_eval.py'),'--output',str(root)],check=True,capture_output=True)
        audit=json.loads((root/'accuracy_audit.json').read_text())
        audit.update(prefill_stages=2,stage2_sources=provenance,stage2_environment=actual_env,
                     stage2_configuration_records=records,address32_routing_records=addr,
                     fresh_math_for_prefill_stage2=True,matched_service_root=str(args.service_root))
        (root/'accuracy_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        subprocess.run(['python3',str(base/'archive_completed_pair.py'),'--single-probe','--source',str(root),
                        '--completion-file','accuracy_audit.json','--label','prefill_stage2_math_20261013'],check=True)
    return code


if __name__=='__main__':raise SystemExit(main())
