# SPDX-License-Identifier: Apache-2.0
"""Recover only absent workload metadata, then run unchanged full validators."""
import hashlib,importlib.util,inspect,json,statistics,subprocess,sys,types
from pathlib import Path
BASE=Path(__file__).resolve().parent
REPO=BASE.parents[1]
ROOT=Path('/workspace/benchmark-results/20261021_nn_silu_fx_official4k_tianshu')
MATH=Path('/workspace/benchmark-results/20261021_nn_silu_fx_math8_tianshu')
OUTPUT=Path('/workspace/benchmark-results/20261021_nn_silu_fx_official_CPU_recovery_tianshu')

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    original=read(ROOT/'result.json');original_hash=sha(ROOT/'result.json')
    assert original['phase']=='audit_failed' and original['error']=="KeyError('workload')"
    assert original['completed_runs']==4 and original['benchmark_exit_code']==0 and 'workload' not in original
    frozen={**original['frozen_sources'],**original['controller_sources']}
    for filename,digest in frozen.items():assert sha(filename)==digest,filename
    directory=REPO/'experiments/tianshu_nn_silu_fx_validation_20261021';sys.path.insert(0,str(directory))
    spec=importlib.util.spec_from_file_location('_FX_official_complete_CPU_recovery',directory/'framework_profile.py')
    framework=importlib.util.module_from_spec(spec);spec.loader.exec_module(framework)
    rows=framework.csv_rows(ROOT,'raw_runs_*.csv')
    assert len(rows)==4 and rows==original['raw_runs']
    expected=[4096,1024,64,256]
    command=original['benchmark_command']
    assert command[command.index('--test-cases')+1]=='[[4096,1024,64,256]]'
    for row in rows:
        assert [int(row[key]) for key in ('Prefill','Decode','Conc','Num Prompts')]==expected
        assert row['Run Status']=='SUCCESS' and int(float(row['Successful Requests']))==256
        assert int(float(row['Total Input Tokens']))==4096*256 and int(float(row['Total Output Tokens']))==1024*256
    workload=dict(input_tokens=4096,output_tokens=1024,concurrency=64,num_prompts=256)
    state=dict(original,workload=workload)
    assert set(state)==set(original)|{'workload'}
    assert all(state[key]==value for key,value in original.items())
    # No validator is patched, no requirement is skipped; legacy input schema alone is restored.
    profile=framework.audit_runtime(ROOT,state,official=True)
    math_state=read(MATH/'result.json');assert math_state['phase']=='complete'
    math_profile=framework.audit_runtime(MATH,math_state,official=False)
    quality=framework.audit_math(MATH,math_state,math_profile,write=False)
    final=framework.audit_benchmark(ROOT,state,profile,quality,write=False)
    assert final['complete'] and final['official_benchmark'] and final['requests_successful']==1024
    assert final['fresh_official_Math8_correct']==101 and final['fresh_C64_Math_correct']==103
    assert final['passes_public_4k_performance_gates']
    assert abs(final['total_tok_s']-5618.01)<.001 and abs(final['ttft_ms']-3835.46)<.001
    # Capture legacy validator source identities and prove they equal the prelaunch Git state.
    paths=set();seen=set()
    def visit(module):
        if id(module) in seen:return
        seen.add(id(module));filename=getattr(module,'__file__',None)
        if filename and Path(filename).resolve().is_relative_to(REPO):paths.add(Path(filename).resolve())
        for value in vars(module).values():
            if isinstance(value,types.ModuleType):
                filename=getattr(value,'__file__',None)
                if filename and Path(filename).resolve().is_relative_to(REPO):visit(value)
            elif inspect.isfunction(value):
                filename=inspect.getsourcefile(value)
                if filename and Path(filename).resolve().is_relative_to(REPO):paths.add(Path(filename).resolve())
    visit(framework)
    audited={}
    for path in paths:
        reference=subprocess.check_output(['git','-C',str(REPO),'show','21f9a0e:'+str(path.relative_to(REPO))])
        assert hashlib.sha256(reference).hexdigest()==sha(path),str(path)
        audited[str(path)]=sha(path)
    assert sha(ROOT/'result.json')==original_hash and read(ROOT/'result.json')['phase']=='audit_failed'
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT/'full_comparison.json').write_text(json.dumps(final,indent=2)+'\n')
    summary={key:value for key,value in final.items() if key!='profile'}
    summary.update(CPU_only_recovery=True,original_controller_phase='audit_failed',original_controller_result_unchanged=True,
        original_failure="KeyError('workload')",only_added_metadata={'workload':workload},
        workload_proven_from_unmodified_benchmark_command_and_all_four_CSV_rows=True,
        all_original_runtime_Math_benchmark_and_C64_validators_executed_unchanged=True,
        legacy_auditor_sources_equal_prelaunch_Git21f9a0e=audited,
        runtime_source_count=len(original['frozen_sources']),controller_source_count=len(original['controller_sources']),
        zero_preemptions=profile['zero_preemptions'],
        vs_previous_formal_TTFT_percent=(final['ttft_ms']/3780.63-1)*100,
        profile_scalar_summary={key:value for key,value in profile.items() if isinstance(value,(str,int,float,bool))},
        original_result_sha256=original_hash,full_comparison_raw_path=str(OUTPUT/'full_comparison.json'),
        full_comparison_sha256=sha(OUTPUT/'full_comparison.json'),recovery_source=str(Path(__file__).resolve()),recovery_source_sha256=sha(__file__))
    (BASE/'OFFICIAL_RECOVERY_GATE_32g.json').write_text(json.dumps(summary,indent=2)+'\n')
    (BASE/'RAW_SHA256_32g.txt').write_text(''.join(sha(path)+'  '+str(path)+'\n' for path in
        [ROOT/'result.json',MATH/'accuracy_audit.json',ROOT/'benchmark.log',*sorted((ROOT/'benchmark_results').glob('*.csv')),OUTPUT/'full_comparison.json']))
    print(json.dumps({key:summary[key] for key in ('complete','CPU_only_recovery','total_tok_s','output_tok_s','ttft_ms','postfirst_spread_percent','vs_previous_formal_percent','vs_previous_formal_TTFT_percent','zero_preemptions','fresh_official_Math8_correct','fresh_C64_Math_correct','runtime_source_count')},indent=2))

if __name__=='__main__':main()
