# SPDX-License-Identifier: Apache-2.0
"""Bind a bit-preserving paired-K decode after qualified S16 installation."""
import atexit,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT_SHA='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'
CANDIDATE_SHA='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'

def select(mode,kwargs):
    return mode=='candidate' and bool(kwargs['IS_3D'])

class RoutedKernel:
    def __init__(self,control,candidate,mode,trace,is_capturing,save,record):
        self.control,self.candidate,self.mode=control,candidate,mode
        self.trace,self.is_capturing,self.save,self.record=trace,is_capturing,save,record
        self.fn=control.fn
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            assert not args
            selected=select(self.mode,kwargs)
            active=self.candidate if selected else self.control
            if selected:
                k=kwargs['key_cache_ptr']
                assert k.data_ptr()%2==0 and all(v%2==0 for v in k.stride()[:-1]) and k.stride(-1)==1
                assert kwargs['BLOCK_SIZE']==16 and kwargs['HEAD_SIZE']==kwargs['HEAD_SIZE_PADDED']==128
                assert kwargs['KV_QUANT_MODE']==0 and not kwargs['USE_TD'] and not kwargs['USE_TD_QO']
                assert str(k.dtype)=='torch.uint8' and str(kwargs['value_cache_ptr'].dtype)=='torch.bfloat16'
            compiled=active[grid](**kwargs)
            key=str((self.trace['active_graph'],self.is_capturing(),bool(kwargs['IS_3D']),kwargs['num_seqs'],kwargs['NUM_SEGMENTS_PER_SEQ'],selected))
            rows=self.record['launches']
            row=rows.setdefault(key,dict(graph_key=self.trace['active_graph'],inside_capture=self.is_capturing(),
                IS_3D=bool(kwargs['IS_3D']),sequences=kwargs['num_seqs'],segments=kwargs['NUM_SEGMENTS_PER_SEQ'],
                selected_pair_rows=selected,active_source_sha256=CANDIDATE_SHA if selected else PARENT_SHA,
                grid=list(grid),tile=kwargs['TILE_SIZE'],warps=compiled.metadata.num_warps,stages=compiled.metadata.num_stages,
                n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),
                kernel_hash=getattr(compiled,'hash',None),calls=0))
            row['calls']+=1
            if not row['inside_capture'] and row['calls']==1:self.save()
            return compiled
        return launch

def install(worker,mode,directory):
    assert mode in ('control','candidate')
    import torch
    import vllm.v1.attention.backends.triton_attn as backend
    from vllm.platforms import current_platform
    assert current_platform.vendor_name=='iluvatar'
    assert worker.vllm_config.speculative_config is None
    producer=backend._soft8_pipeline_installed['producer']
    control=producer.kernel
    parent=Path(inspect.getsourcefile(control.fn))
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==PARENT_SHA
    # S16 changed the JIT function's module while retaining the original
    # Python dispatch module. Locate the object that actually owns Producer.
    candidates={id(module):module for module in tuple(sys.modules.values())
        if module is not None and getattr(module,'__dict__',{}).get('kernel_unified_attention') is producer}
    assert len(candidates)==1
    reader=next(iter(candidates.values()))
    assert reader.kernel_unified_attention is producer and reader.reduce_segments.producer is producer
    original_reducer=reader.reduce_segments.kernel
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    source_path=BASE.parent/'tianshu_k8v16_pair_load_20261020/source.py'
    spec=importlib.util.spec_from_file_location('_service_pair_source',source_path)
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    generated,proof=source.load(parent,directory/'generated_reader','pair_rows')
    assert proof['generated_sha256']==CANDIDATE_SHA
    trace=sys.modules['_tianshu_long_segments_actual_graph_trace'].STATE
    record=dict(installed=True,mode=mode,parent_source=str(parent),parent_sha256=PARENT_SHA,
        candidate_source=str(generated.__file__),candidate_sha256=CANDIDATE_SHA,proof=proof,
        binding_source=str(Path(__file__).resolve()),binding_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dispatch_module_source=str(reader.__file__),dispatch_owner_resolved_by_actual_producer_identity=True,
        launches={},old_S16_binding_record_precedes_this_outer_wrapper=True,
        unchanged_writer_cache_V_values_and_reducer=True,RNG_not_read_or_modified=True,
        arguments_forwarded_unchanged=True,observations_are_Python_launches_not_graph_replay_counts=True)
    def save():
        path=directory/f'binding_{os.getpid()}.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
    producer.kernel=RoutedKernel(control,generated.kernel_unified_attention,mode,trace,torch.cuda.is_current_stream_capturing,save,record)
    assert reader.reduce_segments.kernel is original_reducer
    atexit.register(save);save()
