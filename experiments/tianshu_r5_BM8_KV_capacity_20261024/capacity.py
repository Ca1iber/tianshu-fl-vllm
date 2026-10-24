# SPDX-License-Identifier: Apache-2.0
"""Startup-only native KV allocation quantization, independent of requests."""
import functools,hashlib,json
from pathlib import Path
QUANTUM=256

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def install(module,record_dir):
    original=module.get_kv_cache_configs
    assert not getattr(original,'_tianshu_quantized_capacity',False)
    @functools.wraps(original)
    def get_configs(vllm_config,kv_cache_specs,available_memory):
        assert len(kv_cache_specs)==len(available_memory)==1
        specs=kv_cache_specs[0]
        assert len(specs)==42
        assert all(s.block_size==16 and s.page_size_bytes==12288 for s in specs.values())
        bytes_per_block=sum(s.page_size_bytes for s in specs.values())
        assert bytes_per_block==516096
        supplied=vllm_config.cache_config.num_gpu_blocks_override
        assert supplied is None,'Explicit user override must not be silently replaced'
        natural=available_memory[0]//bytes_per_block
        rounded=(natural//QUANTUM)*QUANTUM
        assert rounded>0 and 0<=natural-rounded<QUANTUM
        vllm_config.cache_config.num_gpu_blocks_override=rounded
        configs=original(vllm_config,kv_cache_specs,available_memory)
        assert isinstance(configs,list) and len(configs)==1
        config=configs[0]
        assert config.num_blocks==rounded and len(config.kv_cache_tensors)==42
        allocated=sum(t.size for t in config.kv_cache_tensors)
        assert allocated==rounded*bytes_per_block<=available_memory[0]
        proof=dict(complete=True,startup_only=True,independent_of_workload=True,quantum=QUANTUM,
            available_memory_bytes=available_memory[0],bytes_per_block=bytes_per_block,
            natural_blocks=natural,rounded_blocks=rounded,discarded_blocks=natural-rounded,
            native_override= vllm_config.cache_config.num_gpu_blocks_override,
            native_config_blocks=[c.num_blocks for c in configs],native_tensor_count=42,
            native_allocated_bytes=allocated,native_function_source=module.__file__,
            native_function_source_sha256=digest(module.__file__),source=str(Path(__file__).resolve()),
            source_sha256=digest(__file__))
        folder=Path(record_dir);folder.mkdir(parents=True,exist_ok=True)
        target=folder/'native_capacity.json'
        assert not target.exists()
        target.write_text(json.dumps(proof,indent=2)+'\n')
        return configs
    get_configs._tianshu_quantized_capacity=True
    module.get_kv_cache_configs=get_configs
