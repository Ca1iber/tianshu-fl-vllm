# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,json
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--log',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();state=json.loads((root/'result.json').read_text());assert state['complete'] and state['phase']=='complete'
    for path,digest in state['source_snapshots'].items():assert sha(path)==sha(root/'source_snapshot'/Path(path).relative_to('/'))==digest
    assert state['cold_config_cache_initial_count']==state['cold_benchmark_initial_count']==0
    assert state['cold_boundary_measured_configurations']>0 and state['cold_boundary_benchmark_hits']==0
    assert state['native_run_mode_and_protocol_unchanged'] and state['actual_entry_classes']==4 and state['actual_ConfigCache_keys']==2
    assert state['original_cache_objects_restored'] and state['original_selected_37_and_64_configs_unchanged'] and state['existing_low_entry_preserved']
    warm=state['prewarm'];assert warm['complete'] and warm['dispatch_classes']==4 and len(warm['calls'])==4 and warm['actual_entry_presence_checked']
    assert {tuple(r['dns_classes'][-2:]) for r in warm['calls']}=={('i32','i32'),('i32','i64'),('i64','i32'),('i64','i64')}
    assert len(state['replay'])==256 and all(r['batch'] in range(1,65) for r in state['replay'])
    data=torch.load(root/'B37_and_RNG_fullwords.pt',map_location='cpu',weights_only=True)
    before,after=data['default_before'],data['default_after'];assert before.shape==after.shape==(37,130560)
    assert torch.equal(before.view(torch.int32),after.view(torch.int32))
    high_diff=int((data['high_before'].view(torch.int32)!=data['high_after'].view(torch.int32)).sum());assert high_diff==state['high_B37_different_words']
    for prefix in ('CPU','GPU'):assert torch.equal(data[prefix+'_before'],data[prefix+'_after'])
    log=args.log.read_text();cold=log.split('CONTROL_COLD37_HIGH_BEGIN',1)[1].split('CONTROL_COLD37_HIGH_END',1)[0]
    assert 'Triton autotuning for function fused_exponential_kernel_f32_unroll8' in cold
    for a,b in [('WARM_HIGH37_BEGIN','WARM_HIGH37_END'),('REPLAY_ALL_CLASSES_BEGIN','REPLAY_ALL_CLASSES_END')]:
        text=log.split(a,1)[1].split(b,1)[0];assert 'Triton autotuning for function' not in text
    gate=dict(complete=True,independently_CPU_rechecked=True,component_only=True,adopted=False,service_not_installed=True,Math_not_run=True,
        controlled_empty_memory_config_and_benchmark_caches=True,native_autotune_protocol_unchanged=True,
        cold_boundary_elapsed_s=state['cold_boundary_elapsed_s'],warm_boundary_elapsed_s=state['warm_boundary_elapsed_s'],prewarm_elapsed_s=warm['elapsed_s'],
        actual_dispatch_classes=4,representative_warm_calls=4,actual_ConfigCache_keys=2,replay_cases=256,replay_autotuning_records=0,
        default_CPU_and_GPU_RNG_exact=True,actual_B37_FP32_words=before.numel(),existing_low_dispatch_preserved=True,
        high_B37_different_words=high_diff,output_identity_scope='Existing low/default entry preserved; high boundary can use a different native block configuration',
        source_snapshots=state['source_snapshots'],raw_root=str(root),raw_result_sha256=sha(root/'result.json'),fixture_sha256=sha(root/'B37_and_RNG_fullwords.pt'),verifier_sha256=sha(__file__))
    (BASE/'GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'CACHE_EVENTS_32g.json').write_text(json.dumps({k:state[k] for k in ('prewarm','memory_configurations','benchmark_pools','original_configs_before','original_configs_after')},indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k!='source_snapshots'}))
if __name__=='__main__':main()
