# SPDX-License-Identifier: Apache-2.0
"""Activate vendor BF16 linear after native warmup; preserve all INT8 branches."""
import atexit,functools,hashlib,inspect,json,os
from pathlib import Path

def install_route(runner_class,banks,dense_record,save_dense):
    import torch,flag_gems
    mode=os.environ['TIANSHU_PREFILL_LINEAR_MODE'];assert mode in ('control','candidate')
    directory=Path(os.environ['TIANSHU_PREFILL_LINEAR_RECORD_DIR']);directory.mkdir(parents=True,exist_ok=False)
    native_capture=runner_class.capture_model
    native_linear=flag_gems.linear
    def identity(fn):
        path=Path(inspect.getsourcefile(fn)).resolve()
        return dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),module=fn.__module__)
    record=dict(installed=True,mode=mode,stage='before_capture',minimum_M=128,
        policy_source=str(Path(__file__).resolve()),policy_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_capture=identity(native_capture),native_linear=identity(native_linear),warmup_GateUp_M2048_prefixes=[],
        activation_count=0,activation_after_qualified_warmup=False,original_BF16_and_packed_weights_unchanged=True,
        input_contiguous_inside_actual_timed_path=True,INT8_M8_M64_not_intercepted=True,RNG_not_read_or_modified=True,calls={})
    warm=set()
    def save():
        path=directory/f'route_{os.getpid()}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
    def dispatch(x,weight,bank,original_mm):
        assert bank['role']=='gateup' and weight.shape==(12288,2048) and x.ndim==2 and x.shape[1]==2048
        m=x.shape[0];capturing=torch.cuda.is_current_stream_capturing();stage=record['stage']
        if stage=='before_capture' and m==2048:warm.add(bank['prefix'])
        use_linear=mode=='candidate' and stage!='before_capture' and m>=128
        implementation='linear_contiguous' if use_linear else 'qualified_mm'
        key=str((bank['prefix'],m,tuple(x.stride()),stage,capturing,implementation))
        row=record['calls'].setdefault(key,dict(prefix=bank['prefix'],bank_id=next(i for i,b in banks.items() if b is bank),
            M=m,N=12288,K=2048,input_stride=list(x.stride()),stage=stage,actually_inside_capture=capturing,
            implementation=implementation,calls=0))
        row['calls']+=1
        result=native_linear(x.contiguous(),weight,None) if use_linear else original_mm(x,weight.T)
        # The last layer persists complete prefix coverage while avoiding per-layer I/O.
        if bank['prefix']=='model.layers.41.mlp.gate_up_proj' and row['calls']&(row['calls']-1)==0:save()
        return result
    @functools.wraps(native_capture)
    def capture(owner,*args,**kwargs):
        assert record['stage']=='before_capture' and len(warm)==42,(record['stage'],len(warm))
        record.update(stage='capturing',warmup_GateUp_M2048_prefixes=sorted(warm),activation_count=1,activation_after_qualified_warmup=True)
        save();save_dense()
        record.update(torch_allocated_before_capture=int(torch.cuda.memory_allocated()),torch_reserved_before_capture=int(torch.cuda.memory_reserved()))
        value=native_capture(owner,*args,**kwargs)
        record.update(torch_allocated_after_capture=int(torch.cuda.memory_allocated()),torch_reserved_after_capture=int(torch.cuda.memory_reserved()),process_peak_allocated_after_capture=int(torch.cuda.max_memory_allocated()))
        record['stage']='ready';save();return value
    runner_class.capture_model=capture
    atexit.register(save);save();return dispatch
