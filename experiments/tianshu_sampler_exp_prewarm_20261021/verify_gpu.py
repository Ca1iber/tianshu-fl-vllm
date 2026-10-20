# SPDX-License-Identifier: Apache-2.0
"""Independent CPU full-word/RNG verification and replay-stage tuning audit."""
import argparse,hashlib,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--log',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();state=json.loads((root/'result.json').read_text());assert state['complete'] and state['phase']=='complete'
    for path,digest in state['source_snapshots'].items():assert sha(path)==sha(root/'source_snapshot'/Path(path).relative_to('/'))==digest
    data=torch.load(root/'RNG_and_B37_fullwords.pt',map_location='cpu',weights_only=True);a,b=data['baseline'],data['actual']
    assert a.shape==b.shape==(37,130560) and a.dtype==b.dtype==torch.float32
    assert torch.equal(a.view(torch.int32),b.view(torch.int32)) and torch.isfinite(a).all() and (a>0).all()
    assert torch.equal(data['baseline_choices'],data['actual_choices']) and torch.equal(a.reciprocal().argmax(-1),data['baseline_choices'])
    for prefix in ('CPU','GPU'):assert torch.equal(data[prefix+'_before'],data[prefix+'_after'])
    warm=state['prewarm'];assert warm['complete'] and len(warm['calls'])==192 and warm['offset_bases']==[0,2**31,2**32]
    for offset in warm['offset_bases']:assert [r['batch'] for r in warm['calls'] if r['offset_base']==offset]==list(range(1,65))
    assert [r['batch'] for r in state['replay']]==list(range(64,0,-1))
    log=args.log.read_text();assert log.count('REPLAY_SWEEP_BEGIN')==log.count('REPLAY_SWEEP_END')==1
    replay_text=log.split('REPLAY_SWEEP_BEGIN',1)[1].split('REPLAY_SWEEP_END',1)[0]
    assert 'Triton autotuning for function' not in replay_text
    gate=dict(complete=True,independently_CPU_rechecked=True,adopted=False,component_only=True,service_not_installed=True,Math_not_run=True,
        default_CPU_and_GPU_RNG_exact=True,actual_B37_FP32_words=a.numel(),actual_B37_choices=37,
        warm_batches=64,warm_calls=192,offset_type_boundaries=[0,2**31,2**32],replay_batches=64,replay_autotuning_records=0,
        prewarm_elapsed_s=warm['elapsed_s'],scratch_bytes=warm['allocation_bytes'],
        scope=state['output_identity_scope'],source_snapshots=state['source_snapshots'],raw_root=str(root),raw_result_sha256=sha(root/'result.json'),
        raw_fixture_sha256=sha(root/'RNG_and_B37_fullwords.pt'),verifier_sha256=sha(__file__))
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'WARMUP_ROUNDS_32g.json').write_text(json.dumps(dict(warmup=warm['calls'],replay=state['replay']),indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k!='source_snapshots'}))
if __name__=='__main__':main()
