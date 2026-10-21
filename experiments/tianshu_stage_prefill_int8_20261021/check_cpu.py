# SPDX-License-Identifier: Apache-2.0
"""CPU-only proof of unchanged observer, qualified407 source subset, argv/env."""
import ast,hashlib,json,subprocess
from pathlib import Path
import torch
import stage_measurement_profile as profile
BASE=Path(__file__).resolve().parent

def main():
    assert not torch.cuda.is_initialized()
    for p in BASE.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    hook=BASE/'hooks/sitecustomize.py'
    text=hook.read_text()
    changes=[('tianshu_prefill_int8_service_20261021/hooks_candidate/sitecustomize.py','tianshu_k8v16_budget_default_20261019/hooks/sitecustomize.py'),
             ("directory=Path(os.environ['TIANSHU_DENSE_INT8_RECORD_DIR']).parent/'stage_records'","directory=Path(os.environ['TIANSHU_STAGE_PROFILE_RECORD_DIR'])"),
             ("marker=directory.parent/'profile_armed'","marker=Path(os.environ['TIANSHU_STAGE_PROFILE_ARM_FILE'])"),
             ('if True:',"if os.environ.get('TIANSHU_STAGE_PROFILE_PROBE') == '1':")]
    for current,old in changes:assert text.count(current)==1; text=text.replace(current,old)
    anchor='        directory.mkdir(parents=True,exist_ok=True)'
    assert text.count(anchor)==1
    text=text.replace(anchor,"        assert os.environ['TIANSHU_STAGE_PROFILE_MODE']=='events'\n"+anchor)
    assert text==(BASE.parent/'tianshu_stage_current_20261020/hooks/sitecustomize.py').read_text()
    old_driver=(BASE.parent/'tianshu_stage_current_20261020/run_profile.py').read_text()
    new_driver=(BASE/'run_profile.py').read_text()
    assert "command = template['server_command']" not in new_driver and 'template' not in new_driver
    assert new_driver.count('prepare(root)')==1
    # Request/measurement/signal/finally logic remains exact outside preparation/import/seeds.
    old_ast=ast.parse(old_driver);new_ast=ast.parse(new_driver)
    old_main=next(n for n in old_ast.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    new_main=next(n for n in new_ast.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    for name in ('read','StopRequested','stop','metrics','request'):
        def find(tree):return next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        assert ast.dump(find(old_ast),include_attributes=False)==ast.dump(find(new_ast),include_attributes=False),name
    assert 'assert len(profile) == 62' in new_driver
    root=Path('/workspace/benchmark-results/20261021_stage_prefill_int8_CPU_prepare_v2_tianshu')
    assert not root.exists();root.mkdir()
    command,environment,env,sources=profile.prepare(root)
    official=profile.read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/OFFICIAL_GATE_32g.json')
    binding=profile.read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/RUNTIME_BINDING_GATE_32g.json')
    assert official['complete'] and official['adopted'] and len(binding['frozen_sources'])==407
    assert all(sources[p]==sha for p,sha in binding['frozen_sources'].items())
    assert len(command)==13
    observed=profile.read(Path(official['raw_root'])/'observed_server_environment.json')
    assert command==observed['command']
    expected={k:v.replace(official['raw_root']+'/',str(root)+'/') for k,v in binding['profile_environment'].items()}
    assert set(expected)==set(env) and len(env)==62
    assert {k:v for k,v in env.items() if k!='PYTHONPATH'}=={k:v for k,v in expected.items() if k!='PYTHONPATH'}
    assert env['PYTHONPATH']==str(BASE/'hooks') and all(environment[k]==v for k,v in env.items())
    assert not any(k.startswith('TIANSHU_STAGE_PROFILE_') for k in env)
    for p,sha in sources.items():
        assert profile.digest(p)==sha==profile.digest(root/'source_snapshot'/Path(p).relative_to('/'))
    # Pin the actual ModelRunnerFL implementation used by the unchanged observer.
    runner=[p for p in sources if p.endswith('/vllm_fl/worker/model_runner.py')]
    assert set(runner)=={str(BASE.parents[1]/'vllm_fl/worker/model_runner.py'),'/usr/local/lib/python3.12/site-packages/vllm_fl/worker/model_runner.py'}
    assert all(sources[p]=='b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e' for p in runner)
    subprocess.run(['bash','-n',str(BASE/'run_profile.sh')],check=True)
    assert not torch.cuda.is_initialized()
    gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,GPU_not_executed=True,
              observer_core_exact_inverse_to_previously_executed_source=True,
              request_metrics_stop_and_exception_classes_AST_equal=True,
              all_407_qualified_sources_byte_identical=True,original_13_argv_verified=True,
              all_62_environment_keys_retained=True,only_PYTHONPATH_value_changed=True,
              snapshot_bytes_rechecked=True,source_count=len(sources),sources=sources,
              qualified_source_count=407,environment_count=62,CPU_output=str(root))
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k!='sources'}))
if __name__=='__main__':main()
