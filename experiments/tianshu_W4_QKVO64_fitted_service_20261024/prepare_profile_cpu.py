# SPDX-License-Identifier: Apache-2.0
"""Bind the 84 existing GPU-qualified payloads; run only when GPU is idle."""
import json,subprocess
from pathlib import Path
import candidate_profile as profile
BASE=Path(__file__).resolve().parent

def main():
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    assert not (BASE/'CPU_GATE_32g.json').exists() and not (BASE/'SELECTED_BANKS_32g.json').exists()
    checkpoint=BASE/'CHECKPOINT_CPU.sha256'
    assert checkpoint.read_text().split()[0]=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
    core=profile.read(BASE/'CORE_CPU_GATE_32g.json');assert core['complete'] and core['scope_checks']==392
    profile.frozen_check(core['sources'])
    banks=[];components={};payload_files={}
    for role in ('O','QKV'):
        cpu,gpu=profile.component_gate(role)
        assert len(cpu['banks'])==42 and {b['layer'] for b in cpu['banks']}==set(range(42))
        for bank in cpu['banks']:
            variant=bank['variants']['fitted'];path=Path(variant['file'])
            assert profile.digest(path)==variant['file_sha256']
            payload_files[str(path)]=variant['file_sha256']
            suffix='qkv_proj' if role=='QKV' else 'o_proj'
            banks.append(dict(prefix=f"model.layers.{bank['layer']}.self_attn.{suffix}",layer=bank['layer'],role=role,
                N=bank['N'],K=bank['K'],original_sha256=bank['original_sha256'],
                **{k:variant[k] for k in ('packed_sha256','scale_sha256','zero_sha256','payload_bytes')}))
        components[role]=dict(independent_FP64_cells=gpu['independent_FP64_cells'],result_sha256=gpu['result_sha256'],
            independent_gate_sha256=profile.digest(profile.COMPONENTS[role]/'INDEPENDENT_CPU_GATE_32g.json'),
            graph_fitted=next(r for r in gpu['timings'] if r['mode']=='graph' and r['variant']=='W4_fitted'))
    assert len({b['prefix'] for b in banks})==84
    assert sum(b['payload_bytes'] for b in banks)==210567168
    selection=dict(complete=True,group_size=128,format='fitted_affine',additional_payload_bytes=210567168,
        no_activation_or_label_calibration=True,checkpoint_not_modified=True,banks=banks)
    (BASE/'SELECTED_BANKS_32g.json').write_text(json.dumps(selection,indent=2)+'\n')
    root=Path('/dev/shm/tianshu_QKVO64_model_profile_CPU_20261024');root.mkdir()
    command,env,settings,sources=profile.prepare(root,'candidate',check_gate=False)
    assert len(command)==13 and len(settings)==64 and len({**settings,'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']})==65
    assert not any('TIANSHU_O64_AFFINE' in k for k in settings)
    assert settings['TIANSHU_QKVO64_FITTED_MODE']=='candidate'
    gate=dict(complete=True,model_ready=True,model_quality_qualified=False,adopted=False,first_model_GPU_not_run=True,
        original_r4_source_count=944,original944_sources_preserved=True,source_count_without_gate=len(sources),
        original_serve_command=command,profile_environment_count=64,actual_environment_count=65,
        all84_selected_component_payload_files_SHA_verified=True,payload_files=payload_files,components=components,
        additional_payload_bytes=210567168,equal_payload_in_control=True,sources=sources,
        checkpoint_SHA256=checkpoint.read_text().split()[0],no_checkpoint_or_structure_change=True)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    command2,env2,settings2,sources2=profile.prepare(root,'candidate')
    assert command2==command and env2==env and settings2==settings
    assert len(sources2)==len(sources)+1
    print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','payload_files')},indent=2))
if __name__=='__main__':main()
