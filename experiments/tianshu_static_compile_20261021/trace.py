# SPDX-License-Identifier: Apache-2.0
"""Observe native exact-range choices during the real model Graph capture."""
import atexit,functools,hashlib,json,os
from pathlib import Path
BACKEND_SHA='6525318ff3ca2e4978e24dd9232a7ae0a02e920e19604b83db6f6043d1f8a54d'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install(worker,mode,directory):
    import torch
    import vllm.compilation.piecewise_backend as native
    from vllm_fl.worker.model_runner import ModelRunnerFL
    cfg=worker.vllm_config
    assert cfg.model_config.model=='/workspace/MiniCPM5-2B' and cfg.model_config.quantization is None and cfg.model_config.dtype==torch.bfloat16
    assert cfg.speculative_config is None and cfg.lora_config is None and cfg.scheduler_config.max_num_batched_tokens==4096 and cfg.scheduler_config.max_num_seqs==64
    assert sorted(cfg.compilation_config.compile_sizes)==([] if mode=='control' else [8,64])
    assert sha(native.__file__)==BACKEND_SHA
    cls=native.PiecewiseBackend;assert not getattr(cls,'_tianshu_static_range_trace',False)
    record=dict(mode=mode,installed=True,capture_completed=False,stage='installed',actual_compile_sizes=cfg.compilation_config.compile_sizes,
        native_backend_source=str(Path(native.__file__).resolve()),native_backend_sha256=BACKEND_SHA,
        trace_source=str(Path(__file__).resolve()),trace_sha256=sha(__file__),routes={},
        original_range_entry_identity_preserved=True,original_find_called_once=True,RNG_and_inputs_not_read_or_modified=True,
        Python_range_selection_calls_are_not_Graph_replays=True,timing_path_has_no_JSON_writes=True)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    def save():
        path=directory/f'backend_{os.getpid()}.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
    original_find=cls._find_range_for_shape
    @functools.wraps(original_find)
    def find(owner,runtime_shape):
        entry=original_find(owner,runtime_shape)
        if entry is not None:
            trace=getattr(__import__('sys').modules.get('_tianshu_long_segments_actual_graph_trace'),'STATE',{})
            capturing=torch.cuda.is_current_stream_capturing();key=str((owner.piecewise_compile_index,runtime_shape,capturing,trace.get('active_graph'),entry.compile_range.start,entry.compile_range.end))
            row=record['routes'].setdefault(key,dict(piece_index=owner.piecewise_compile_index,runtime_shape=int(runtime_shape),
                selected_range=[entry.compile_range.start,entry.compile_range.end],selected_is_single=entry.compile_range.is_single_size(),
                compiled=entry.compiled,actually_inside_capture=capturing,graph_key=trace.get('active_graph'),
                compile_sizes=list(owner.compile_sizes),range_entries=[[r.start,r.end] for r in owner.range_entries],calls=0))
            row['calls']+=1
        return entry
    cls._find_range_for_shape=find;cls._tianshu_static_range_trace=True
    original_capture=ModelRunnerFL.capture_model
    @functools.wraps(original_capture)
    def capture(owner,*args,**kwargs):
        record['stage']='capture_in_progress';answer=original_capture(owner,*args,**kwargs)
        for m in (8,64):
            rows=[r for r in record['routes'].values() if r['runtime_shape']==m and r['actually_inside_capture']]
            assert rows and all(r['compiled'] and r['graph_key'] is not None for r in rows)
            assert all(r['selected_range']==([m,m] if mode=='candidate' else [1,4096]) for r in rows)
        record.update(stage='active',capture_completed=True,after_original_capture=True);save();return answer
    ModelRunnerFL.capture_model=capture;atexit.register(save);save()
