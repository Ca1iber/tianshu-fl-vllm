# SPDX-License-Identifier: Apache-2.0
"""Warm observed index32/4001 before native capture; no request-time interception."""
import atexit,functools,hashlib,json,os
from pathlib import Path
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install(worker,mode,directory):
    import torch
    import importlib
    index=importlib.import_module('flag_gems.ops.index')
    from vllm_fl.worker.model_runner import ModelRunnerFL
    from flag_gems.utils.code_cache import code_cache_dir
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    record=dict(mode=mode,installed=True,completed=False,rows=[],policy_source=str(BASE/'index_warm.py'),policy_sha256=sha(__file__),index_source=index.__file__,index_sha256=sha(index.__file__),only_startup_prewarms=True,no_model_weight_KV_token_or_request_access=True,request_time_index_not_wrapped=True)
    def save():
        path=directory/f'warm_{os.getpid()}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
    original_capture=ModelRunnerFL.capture_model
    @functools.wraps(original_capture)
    def capture(owner,*args,**kwargs):
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        with torch.inference_mode():
            for length in (63,64):
                source=(torch.arange(length,dtype=torch.int32)*3+7)
                for m in (32,4001):
                    indices=torch.arange(m,dtype=torch.int64)%length;expected=source[indices]
                    actual_input=source.to(owner.device);actual_indices=indices.to(owner.device)
                    output=index.index(actual_input,[actual_indices]);torch.cuda.synchronize()
                    assert torch.equal(output.cpu(),expected)
                    key=index._linearized_adjacent_index_func.arg_key(actual_input,0,[actual_indices])
                    assert key in index._linearized_adjacent_index_func.overloads
                    generated=code_cache_dir()/('index_linearized_'+key+'.py');assert generated.is_file()
                    record['rows'].append(dict(input_shape=[length],input_dtype='torch.int32',index_shape=[m],index_dtype='torch.int64',output_dtype=str(output.dtype),all_outputs_CPU_exact=True,linearized_overload_key=key,generated_source=str(generated),generated_sha256=sha(generated),before_original_capture=True))
                    del actual_input,actual_indices,output
        assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
        record['CPU_GPU_RNG_unchanged']=True;save()
        value=original_capture(owner,*args,**kwargs)
        record['completed']=True;save();return value
    ModelRunnerFL.capture_model=capture;atexit.register(save);save()
