# SPDX-License-Identifier: Apache-2.0
"""Bind only the independently checked long-context kernels at Worker initialization."""
import atexit,functools,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
ORIGINAL='dc69f525d8c7686a8977f50e901af39820ce92486894890631c53fc01dc46700'
CANDIDATE='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106'

def select(original,selected):
    @functools.wraps(original)
    def choose(batch,length):
        configuration=original(batch,length)
        if configuration is not None and batch in selected:
            configuration=dict(configuration,segments=16)
        return configuration
    return choose

def bind(reader,producer,generated,legacy,selected):
    assert reader.kernel_unified_attention is producer
    assert reader.reduce_segments.producer is producer
    original=legacy.choose_config
    replacement=select(original,selected)
    producer.kernel=generated.kernel_unified_attention
    reader.reduce_segments.kernel=generated.reduce_segments
    legacy.choose_config=replacement

def install(worker,mode,directory):
    assert mode in ('control','candidate')
    import torch
    from vllm.platforms import current_platform
    import vllm.v1.attention.backends.triton_attn as backend
    from vllm_fl.worker import model_runner
    import vllm_fl.compilation.graph as graph
    assert current_platform.vendor_name=='iluvatar'
    cfg=worker.vllm_config;hf=cfg.model_config.hf_config
    assert cfg.model_config.dtype==torch.bfloat16 and cfg.model_config.quantization is None
    assert cfg.speculative_config is None and cfg.lora_config is None
    assert (hf.hidden_size,hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads)==(2048,42,16,2)
    assert cfg.parallel_config.tensor_parallel_size==cfg.parallel_config.pipeline_parallel_size==cfg.parallel_config.data_parallel_size==1
    assert cfg.scheduler_config.max_num_batched_tokens=={'control':4096,'candidate':4352}[os.environ['TIANSHU_DENSE_BUDGET_MODE']]
    policy_path=BASE/'batch_policy.json'
    policy=json.loads(policy_path.read_text());selected=set(policy['selected_batches'])
    assert selected==set(range(33,57)) and policy['maximum_segments']==16 and policy['short_context_threshold']==8192
    producer=backend._soft8_pipeline_installed['producer']
    reader=sys.modules[producer.kernel.fn.__module__]
    legacy=sys.modules[type(producer).__module__]
    parent=Path(inspect.getsourcefile(producer.kernel.fn))
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==ORIGINAL
    assert Path(legacy.__file__).resolve()==BASE.parent/'tianshu_k8v16_pipeline_20261019/legacy_routing.py'
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if mode=='candidate':
        source_path=BASE.parent/'tianshu_long_segments_20261020/source.py'
        spec=importlib.util.spec_from_file_location('_tianshu_service_dynamic_segments_source',source_path)
        source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
        generated,proof=source.load(parent,directory/'generated_reader')
        assert proof['generated_sha256']==CANDIDATE
        bind(reader,producer,generated,legacy,selected)
    else:proof=dict(parent_source=str(parent),parent_sha256=ORIGINAL,generated_sha256=ORIGINAL)
    actual_path=Path(inspect.getsourcefile(producer.kernel.fn))
    reducer_path=Path(inspect.getsourcefile(reader.reduce_segments.kernel.fn))
    expected=CANDIDATE if mode=='candidate' else ORIGINAL
    assert hashlib.sha256(actual_path.read_bytes()).hexdigest()==expected
    assert hashlib.sha256(reducer_path.read_bytes()).hexdigest()==expected
    record=dict(mode=mode,installed=True,policy=policy,policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                producer_source=str(actual_path),reducer_source=str(reducer_path),actual_kernel_sha256=expected,
                proof=proof,worker_initialization_completed_before_install=True,
                unchanged_reader_python_dispatch=True,KV_format_writer_and_model_parameters_unchanged=True,
                RNG_not_read_or_modified=True,Graph_trace_source=str(BASE.parent/'tianshu_k8v16_graph_segments_v3_20261019/trace.py'))
    def save():
        record['producer_records']=producer.records
        p=directory/f'bindings_{os.getpid()}.json'
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(p)
    atexit.register(save);save()
    path=BASE.parent/'tianshu_k8v16_graph_segments_v3_20261019/trace.py'
    spec=importlib.util.spec_from_file_location('_tianshu_long_segments_actual_graph_trace',path)
    trace=importlib.util.module_from_spec(spec);sys.modules[spec.name]=trace;spec.loader.exec_module(trace)
    trace.install_graph(graph);trace.install_runner(model_runner);trace.install_routing(legacy)
