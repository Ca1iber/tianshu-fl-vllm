# SPDX-License-Identifier: Apache-2.0
"""Recheck preserved Math; resume original GPU controller with explicit auxiliary audit."""
import hashlib,importlib.util,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
PIPELINE=BASE.parent/'tianshu_dense_budget_validation_v2_20261021'
MATH=Path('/workspace/benchmark-results/20261021_dense_budget_v2_math_tianshu')
sys.path.insert(0,str(PIPELINE))
import framework_profile as native_profile

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def checked_audit():
    original=(PIPELINE/'audit_profile.py').read_text();changed=(BASE/'corrected_audit.py').read_text()
    edits=[('BASE=Path(__file__).resolve().parent',"BASE=Path('/workspace/vllm-plugin-FL/experiments/tianshu_dense_budget_validation_v2_20261021')"),
           ("row[prefix+'_sha256']","row['native_source_sha256' if prefix=='native' else 'policy_sha256']")]
    restored=changed
    for before,after in reversed(edits):assert restored.count(after)==1;restored=restored.replace(after,before)
    assert restored==original
    return load('_budget_exact_metadata_key_audit_recovery',BASE/'corrected_audit.py')

def recover_math():
    state=read(MATH/'result.json');assert state['phase']=='audit_failed' and state['error']=="KeyError('native_sha256')" and state['evalscope_exit_code']==0
    assert len(state['frozen_sources'])==374 and len(state['controller_sources'])==9
    native_profile.frozen_check(state['frozen_sources']);native_profile.frozen_check(state['controller_sources'])
    runtime=checked_audit().audit_runtime(MATH,state,official=False)
    quality=native_profile.audit_math(MATH,state,runtime,write=False)
    assert quality['complete'] and quality['total']==105 and quality['correct']==102 and quality['accuracy']>.95
    sources={str(p):sha(p) for p in (BASE/'recover.py',BASE/'corrected_audit.py')}
    manifest=dict(complete=True,postprocessing_only=True,GPU_Math_not_rerun=True,original_phase=state['phase'],original_error=state['error'],
        original_result_sha256=sha(MATH/'result.json'),original_state_sha256=sha(MATH/'state.json'),
        exact_two_edits=['dependency BASE remains original immutable pipeline','read actual native_source_sha256 field instead of missing native_sha256'],
        runtime_sources_374_unchanged=True,original_controller_sources_9_unchanged=True,supplemental_controller_sources=sources,
        correct=quality['correct'],total=105,accuracy=quality['accuracy'])
    result=dict(state,phase='complete',profile_audit=runtime,correct=quality['correct'],total=105,accuracy=quality['accuracy'],accuracy_passed=True,error=None,
        postprocessing_recovery=manifest,original_failed_result_retained=True)
    write(MATH/'rechecked_accuracy_audit.json',quality);write(MATH/'rechecked_result.json',result);write(MATH/'postprocessing_recovery.json',manifest)
    print(json.dumps(manifest),flush=True);return quality,result

def run_official():
    quality,result=recover_math();driver=load('_budget_original_official_driver_recovered_audit',PIPELINE/'run_validation.py')
    native_read=driver.read;native_prepare=driver.prepare
    def recovered_read(path):
        path=Path(path)
        if path==MATH/'result.json':return result
        if path==MATH/'accuracy_audit.json':return quality
        return native_read(path)
    def prepare(root,screen_root):
        answer=native_prepare(root,screen_root)
        manifest=read(MATH/'postprocessing_recovery.json');manifest['primary_controller_sources']=read(root/'controller_sources.json')
        for path,digest in manifest['supplemental_controller_sources'].items():
            assert sha(path)==digest;target=root/'supplemental_source_snapshot'/Path(path).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());assert sha(target)==digest
        write(root/'supplemental_controller_sources.json',manifest);return answer
    driver.read=recovered_read;driver.prepare=prepare;driver.audit_runtime=checked_audit().audit_runtime
    return driver.main()
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='recover-math':recover_math()
    else:raise SystemExit(run_official())
