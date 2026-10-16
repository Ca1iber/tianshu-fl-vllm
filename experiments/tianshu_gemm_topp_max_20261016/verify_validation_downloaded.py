# SPDX-License-Identifier: Apache-2.0
"""Audit fresh candidate Math105 or original4K from all downloaded raw artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
from validation_profile_new import audit_runtime,audit_math,audit_benchmark,read


def verify_sources(root,state):
    repo=Path(__file__).resolve().parents[2]
    assert len(state['frozen_sources'])==57 and state['mode']=='candidate'
    for source,sha in state['frozen_sources'].items():
        snapshot=root/'source_snapshot'/Path(source).relative_to('/')
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==sha
        if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
            assert hashlib.sha256((repo/Path(source).relative_to('/workspace/vllm-plugin-FL')).read_bytes()).hexdigest()==sha


def verify_math(root):
    state=read(root/'result.json');assert state['phase']=='complete' and state['stage']=='math'
    assert state['fresh_math_for_original_command_profile'] and not state['reused_original_accuracy']
    verify_sources(root,state);profile=audit_runtime(root,state,official=False,online=False)
    assert profile['strictTopP32Kmax'] and profile['M2048_three_projection_tiles'] and profile['original_sampler']
    quality=audit_math(root,state,profile,write=False);assert quality==read(root/'accuracy_audit.json')
    assert state['correct']==quality['correct'] and state['accuracy']==quality['accuracy']
    return state,quality


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--math-root',type=Path);parser.add_argument('--screen-root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();state=read(root/'result.json')
    confirm=read(args.screen_root/'comparison.json');candidate=read(args.screen_root/'candidate/result.json')
    assert confirm['complete'] and confirm['gain_percent']>=2 and confirm['rounds_per_mode']==3 and confirm['excluded_rounds']==[1]
    assert candidate['frozen_sources']==confirm['frozen_sources']
    for path,sha in candidate['frozen_sources'].items():assert state['frozen_sources'][path]==sha
    assert state['screen_root']==str(Path(candidate['output_root']).parent)
    if state['stage']=='math':
        state,quality=verify_math(root)
        result=dict(complete=True,stage='math',correct=quality['correct'],total=105,accuracy=quality['accuracy'],
                    accuracy_requirement_passed=quality['official_requirement_passed'],all105_raw_predictions_and_reviews_verified=True,
                    all57_source_snapshots_verified=True,actual38_environment_and_default75warm_profile_verified=True,
                    fresh_Math=True,reused_old_accuracy=False,incorrect_indices=quality['incorrect_indices'],
                    generation_stop_reasons=quality['generation_stop_reasons'],max_tokens_indices=quality['max_tokens_indices'])
    else:
        assert state['stage']=='official' and state['phase']=='complete' and args.math_root is not None
        _,quality=verify_math(args.math_root.resolve());verify_sources(root,state)
        assert quality['accuracy']>.95 and quality['frozen_sources']==state['frozen_sources']
        profile=audit_runtime(root,state,official=True,online=False)
        comparison=audit_benchmark(root,state,profile,quality,write=False);assert comparison==read(root/'comparison.json')
        result=dict(complete=True,stage='official',all1024_requests_successful=True,all57_source_snapshots_verified=True,
                    fresh_Math_correct=quality['correct'],fresh_Math_total=105,original_serve_argv_and_benchmark_SHA_verified=True,
                    comparison=comparison)
    (root/'local_download_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
