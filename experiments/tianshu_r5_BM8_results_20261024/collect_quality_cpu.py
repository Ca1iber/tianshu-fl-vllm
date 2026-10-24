# SPDX-License-Identifier: Apache-2.0
"""Collect only a completed joint-model quality run, never partial scores."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
MODEL=BASE.parent/'tianshu_r5_BM8_KV_20261024'
sys.path.insert(0,str(MODEL))
from audit import audit_model,audit_math
from candidate_profile import read,digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=int,choices=(8,64),required=True)
    args=parser.parse_args();batch=args.batch
    root=Path(f'/dev/shm/tianshu_r5_BM8_runs_20261024/math{batch}')
    state=read(root/'result.json');assert state['phase']=='complete' and state['evaluation_exit_code']==0 and state['batch']==batch
    q=audit_math(root,state);assert q==read(root/'accuracy_audit.json')
    profile=audit_model(root,state,runtime=True)
    assert json.loads(json.dumps(profile))==read(root/'profile_audit.json')
    assert len(state['frozen_sources'])==1023 and len(state['full_environment'])==67 and len(state['server_command'])==13
    report_path=next(root.glob('evalscope/*/reports/minicpm/math_500.json'));run=report_path.parents[2]
    review=run/'reviews/minicpm/math_500_Level 3.jsonl';pred=run/'predictions/minicpm/math_500_Level 3.jsonl'
    reviews=[json.loads(s) for s in review.read_text().splitlines()]
    predictions={p['index']:p for p in (json.loads(s) for s in pred.read_text().splitlines())}
    assert len(reviews)==len(predictions)==105
    rows=[]
    for r in sorted(reviews,key=lambda r:r['index']):
        p=predictions[r['index']]['model_output'];score=r['sample_score']['score']
        assert score['status']=='success' and p['error'] is None
        rows.append(dict(index=r['index'],accuracy=score['value']['accuracy'],stop_reason=p['choices'][0]['stop_reason'],error=False))
    assert sum(r['accuracy'] for r in rows)==q['correct']
    with (BASE/f'C{batch}_KEY_SCORES_32g.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    compact={k:profile[k] for k in ('complete','additional_payload_bytes','all84_original_and_packed_weight_hashes_verified',
        'QKVO64_capture_banks','actual_FULL64_replays','physical_KV_blocks','model_memory_usage_bytes','model_unique_storage_bytes')}
    gate=dict(complete=True,batch=batch,correct=q['correct'],total=105,accuracy=q['accuracy'],this_batch_above95=q['above95'],
        all105_and_caps_counted=True,request_errors=0,stop_reasons=q['stop_reasons'],incorrect_indices=q['incorrect_indices'],
        source_count=1023,environment_count=67,original_serve_command=state['server_command'],profile=compact,
        raw_root=str(root),raw_files={str(p):digest(p) for p in (root/'result.json',root/'accuracy_audit.json',root/'profile_audit.json',review,pred,report_path)},
        collector_source_sha256=digest(Path(__file__)),official_benchmark=False,adopted=False)
    (BASE/f'C{batch}_QUALITY_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k!='raw_files'},indent=2))
if __name__=='__main__':main()
