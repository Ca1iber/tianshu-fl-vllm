# SPDX-License-Identifier: Apache-2.0
import importlib.util,sys
from candidate_profile import BASE,EXP,digest,read
spec=importlib.util.spec_from_file_location('_BM8_capacity_parent_audit',EXP/'tianshu_r5_BM8_KV_20261024/audit.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
audit_math=parent.audit_math

def audit_model(root,state,runtime=False):
    result=parent.audit_model(root,state,runtime=runtime)
    proof=read(root/'capacity_records/native_capacity.json')
    assert proof['complete'] and proof['startup_only'] and proof['independent_of_workload']
    assert proof['quantum']==256 and proof['rounded_blocks']==proof['native_override']==36096
    assert proof['native_config_blocks']==[36096] and proof['native_tensor_count']==42
    assert proof['native_allocated_bytes']==36096*516096
    assert proof['natural_blocks']//256*256==36096 and 0<=proof['discarded_blocks']<256
    for key in ('source','native_function_source'):
        assert proof[key+'_sha256']==digest(proof[key])==state['frozen_sources'][proof[key]]
    assert result['physical_KV_blocks']==36096
    assert state['full_environment']['VLLM_CACHE_ROOT']==str(root/'vllm_cache')
    result.update(native_capacity_quantized=True,capacity_proof=proof)
    return result
