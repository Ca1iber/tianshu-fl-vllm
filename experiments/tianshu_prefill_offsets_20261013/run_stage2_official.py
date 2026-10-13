# SPDX-License-Identifier: Apache-2.0
"""Original workloads with matched fresh Prefill-stage2/Decode32 quality."""
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
    service=json.loads((args.service_root/'candidate/result.json').read_text());assert service['prefill_stages']==2
    comparison=json.loads((args.service_root/'comparison.json').read_text());assert comparison['complete'] and comparison['gain_percent']>=2
    quality=json.loads((args.math_results/'accuracy_audit.json').read_text())
    assert quality['fresh_math_for_prefill_stage2'] and quality['prefill_stages']==2 and quality['official_requirement_passed']
    base=Path(__file__).resolve().parents[1]
    script='run_official_4k.py' if args.input_tokens==4096 else 'run_official_16k.py'
    trial=base/'tianshu_native_prefill_20261012'/script
    hook=Path(__file__).resolve().parent/'hooks_stage2/sitecustomize.py'
    sources={'stage2_sitecustomize.py':hook,
             'address32_sitecustomize.py':base/'tianshu_attention_offsets_20261013/hooks/sitecustomize.py',
             'address32.py':base/'tianshu_attention_offsets_20261013/address32.py'}
    snapshot=root/'stage2_source_snapshot';snapshot.mkdir();provenance={}
    for name,path in sources.items():
        raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        expected=service['prefill_stage2_hook']['sha256'] if name=='stage2_sitecustomize.py' else service['address32_sources'][name]['sha256']
        assert sha==expected==quality['stage2_sources'][name]['sha256']
        (snapshot/name).write_bytes(raw);provenance[name]=dict(path=str(path),sha256=sha)
    for p in [Path(__file__),trial]:(snapshot/p.name).write_bytes(p.read_bytes())
    actual_env=dict(PYTHONPATH=str(hook.parent),TIANSHU_ADDRESS32_PROBE='1',TIANSHU_ADDRESS32_RECORD_DIR=str(root/'address32_records'),
                    TIANSHU_PREFILL_STAGE2_PROBE='1',TIANSHU_PREFILL_STAGE2_RECORD_DIR=str(root/'prefill_stage2_records'))
    original=subprocess.Popen
    def popen(command,*positional,**kwargs):
        if isinstance(command,list) and command[:2]==['vllm','serve']:
            env=dict(kwargs['env']);env.update(actual_env);kwargs['env']=env
        return original(command,*positional,**kwargs)
    spec=importlib.util.spec_from_file_location('_tianshu_stage2_official_trial',trial)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);subprocess.Popen=popen
    sys.argv=[str(trial),'--output',str(root),'--control-trial',str(args.service_root/'control'),
              '--candidate-trial',str(args.service_root/'candidate'),'--math-results',str(args.math_results),
              '--max-batched-tokens','2048','--decode-launch','1','--buckets','1','--gather','1','--bucket-trace','2']
    try:code=module.main()
    finally:subprocess.Popen=original
    state=json.loads((root/'result.json').read_text())
    configured=[json.loads(f.read_text()) for f in (root/'prefill_stage2_records').glob('configured_*.json')]
    addr=[json.loads(f.read_text()) for f in (root/'address32_records').glob('route_*.json')]
    state.update(prefill_stages=2,stage2_sources=provenance,stage2_environment=actual_env,
                 stage2_configuration_records=configured,address32_routing_records=addr,
                 matched_service_root=str(args.service_root),matched_math_root=str(args.math_results))
    (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    if code==0:
        assert configured and addr and all(r['configuration']['num_stages']==2 for r in configured)
        assert all(r['configuration']==dict(BLOCK_M=256,TILE_SIZE=128,num_warps=16,num_stages=2) for r in state['native_prefill_routing_records'])
        assert {r['generated_sha256'] for r in addr}=={r['generated_sha256'] for r in quality['address32_routing_records']}
        audit_script='audit_official.py' if args.input_tokens==4096 else 'audit_official_16k.py'
        previous=('3202.05','4501.53') if args.input_tokens==4096 else ('2600.48','189964.79')
        subprocess.run(['python3',str(base/'tianshu_native_prefill_20261012'/audit_script),'--output',str(root),
                        '--math-results',str(args.math_results),'--previous-best-tok-s',previous[0],
                        '--previous-best-ttft-ms',previous[1]],check=True,capture_output=True)
        result=json.loads((root/'comparison.json').read_text())
        result.update(prefill_stages=2,stage2_sources=provenance,stage2_configuration_records=configured,
                      address32_routing_records=addr,fresh_math_for_prefill_stage2=True)
        result['profile']=result['profile'].replace('Prefill M256/KV128/w16/s1','Prefill M256/KV128/w16/s2')+' +guarded int32 Decode offsets'
        (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        subprocess.run(['python3',str(base/'archive_completed_pair.py'),'--single-probe','--source',str(root),
                        '--label',f'prefill_stage2_official{args.input_tokens}_20261013'],check=True)
    return code


if __name__=='__main__':raise SystemExit(main())
