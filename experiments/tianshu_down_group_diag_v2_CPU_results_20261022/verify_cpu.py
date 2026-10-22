# SPDX-License-Identifier: Apache-2.0
import ctypes,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=Path('/workspace/benchmark-results/20261022_down_group_diag_v2_GPU_tianshu');FAILED=Path('/workspace/benchmark-results/20261022_down_group_GPU_tianshu');MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def f32(a):return (np.asarray(a,np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16(a):
 u=np.asarray(a,np.float32).view(np.uint32);return ((u+32767+((u>>16)&1))>>16).astype(np.uint16)
def main():
 assert not (BASE/'CPU_FMA_GATE_32g.json').exists();state=read(ROOT/'result.json');failed=read(FAILED/'result.json');assert state['complete'] and state['phase']=='diagnostic_complete' and state['candidate_not_qualified'] and failed['phase']=='failed' and failed['error']=='AssertionError()' and sha(FAILED/'result.json')==state['original_failed_result_sha256'] and state['checkpoint_before']==state['checkpoint_after']=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
 for p,h in state['frozen_sources'].items():assert sha(p)==h and sha(ROOT/'source_snapshot'/Path(p).relative_to('/'))==h
 with MODEL.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
 info=header['model.layers.0.mlp.down_proj.weight'];words=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+length+info['data_offsets'][0],shape=(2048,6144));weight=f32(words);scales=np.asarray(np.maximum(abs(weight).max(1),np.float32(1.e-8)).astype(np.float64)/127.,np.float32);packed=np.clip(np.rint(np.asarray(weight.astype(np.float64)/scales[:,None].astype(np.float64),np.float32)),-127,127).astype(np.int8);assert ahash(words)==failed['bank']['Down_original_BF16_sha256'] and ahash(packed)==failed['bank']['Down_packed_sha256'] and ahash(scales)==failed['bank']['Down_scales_sha256']
 libm=ctypes.CDLL('libm.so.6');fmaf=libm.fmaf;fmaf.argtypes=[ctypes.c_float]*3;fmaf.restype=ctypes.c_float;summary=[];dot_checks=projection_checks=0
 for group in (128,512):
  cases=[c for c in state['cases'] if c['group']==group];assert len(cases)==2 and all(c['native_INT32_full_CPU_exact'] and c['native_reference_vs_CPU_separate_BF16_mismatches']==0 and c['original_vs_CPU_fma_BF16_mismatches']==0 for c in cases);first=cases[0];assert sha(ROOT/first['base_artifact'])==first['base_sha256']
  with np.load(ROOT/first['base_artifact'],allow_pickle=False) as data:
   q=data['Aq'];asc=data['AS'];native=data['native_integer_groups'];cpuinteger=data['CPU_integer_groups'];cpu_fma=data['CPU_fma_acc'];separate_words=data['CPU_separate_words'];fma_words=data['CPU_fma_words'];assert np.array_equal(native,cpuinteger)
   debug=first['instrumented'][0];assert sha(ROOT/debug['artifact'])==debug['artifact_sha256']
   with np.load(ROOT/debug['artifact'],allow_pickle=False) as instrumented:original=instrumented['original_words'].copy()
   affected=np.flatnonzero(original!=separate_words);selected=np.unique(np.concatenate([affected,np.random.default_rng(202610227800+group).integers(0,64*2048,size=48)]))
   for offset in selected:
    row,col=divmod(int(offset),2048);acc=np.float32(0.)
    for index,start in enumerate(range(0,6144,group)):
     integer=int(q[row,start:start+group].astype(np.int64)@packed[col,start:start+group].astype(np.int64));assert integer==int(native[index,row,col]);acc=np.float32(fmaf(float(integer),float(asc[row,index]),float(acc)));dot_checks+=1
    assert acc==cpu_fma[row,col];projected=bf16(np.asarray([np.float32(acc*scales[col])],np.float32))[0];assert projected==original[row,col]==fma_words[row,col];projection_checks+=1
  assembly=ROOT/f'G{group}_w2_original_asm.txt';assert 'ml_fma_f32' in assembly.read_text();rn_assembly=ROOT/f'G{group}_w2_RN1_asm.txt';assert 'ml_fma_f32' in rn_assembly.read_text()
  for case in cases:
   assert case['original_compiled']['binary_sha256']==failed['cases'][0]['compiled'][f'G{group}_w{case["warps"]}']['binary_sha256']
   for debug in case['instrumented']:assert debug['debug_INT32_full_CPU_exact'] and debug['BF16_mismatches_vs_original']==0 and debug['BF16_mismatches_vs_CPU_fma']==0 and debug['FP32_acc_mismatches_vs_CPU_fma']==0 and sha(ROOT/debug['artifact'])==debug['artifact_sha256']
  summary.append(dict(group=group,outputs=64*2048,BF16_mismatches_vs_separate_rounding=int(affected.size),BF16_mismatches_vs_FMA=0,all_native_and_instrumented_INT32_exact=True,both_warps_identical=True,explicit_mul_add_RN_still_lowered_to_device_FMA=True,libc_fmaf_selected_outputs=len(selected),original_assembly_sha256=sha(assembly),explicit_RN_assembly_sha256=sha(rn_assembly)))
 result=dict(complete=True,CPU_only=True,no_GPU_imports=True,diagnostic_only=True,original_experiment_passed=False,no_original_source_or_raw_modified=True,posthoc_CPU_phase_separate_from_frozen_GPU_controller=True,all_source_snapshots_rechecked=True,original_compiled_binaries_reproduced=True,independent_INT64_group_dot_checks=dot_checks,independent_libc_fmaf_projection_checks=projection_checks,device_FMA_confirmed_by_assembly_and_outputs=True,groups=summary,diagnostic_GPU_RNG_guards_not_claimed=True,model_quality_not_tested=True,timing_not_performed=True,adopted=False,raw_root=str(ROOT),raw_result_sha256=sha(ROOT/'result.json'),verifier_source_sha256=sha(__file__))
 (BASE/'CPU_FMA_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps(result))
if __name__=='__main__':main()
