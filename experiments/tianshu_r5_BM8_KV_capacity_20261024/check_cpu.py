# SPDX-License-Identifier: Apache-2.0
import ast,json,types
from pathlib import Path
from candidate_profile import BASE,prepare,digest,frozen_check,released_sources
from capacity import install

def main():
    root=Path('/dev/shm/tianshu_BM8_capacity_CPU_20261024')
    assert not root.exists();root.mkdir()
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.core import kv_cache_utils as native
    import torch
    config=EngineArgs(model='/workspace/MiniCPM5-2B',max_model_len=131072,max_num_seqs=64,max_num_batched_tokens=4096).create_engine_config()
    proofs=[]
    for blocks in (36283,36291):
        proxy=types.SimpleNamespace(get_kv_cache_configs=native.get_kv_cache_configs,__file__=native.__file__)
        install(proxy,root/f'native_{blocks}')
        config.cache_config.num_gpu_blocks_override=None
        specs={f'model.layers.{i}.self_attn.attn':FullAttentionSpec(block_size=16,num_kv_heads=2,head_size=128,dtype=torch.uint8,page_size_padded=12288) for i in range(42)}
        configs=proxy.get_kv_cache_configs(config,[specs],[blocks*516096])
        assert configs[0].num_blocks==36096
        assert sum(t.size for t in configs[0].kv_cache_tensors)==36096*516096
        proofs.append(json.loads((root/f'native_{blocks}'/'native_capacity.json').read_text()))
    command,env,profile,sources=prepare(root,'candidate',check_gate=False)
    assert len(command)==13 and len(profile)+1==70
    for path in BASE.rglob('*.py'):ast.parse(path.read_text(),filename=str(path))
    released=released_sources();assert all(sources[p]==h for p,h in released.items())
    frozen_check(sources)
    result=dict(complete=True,model_ready=True,CPU_only=True,native_CPU_allocation_verified=True,
        native_CPU_cases=proofs,original997_exact_subset=True,source_count_without_self=len(sources),sources=sources,
        source_snapshot_directory=str(root/'source_snapshot'),candidate_profile_environment=profile,
        original_serve_argv=command,fresh_C8_C64_quality_required=True,adopted=False)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','model_ready','native_CPU_allocation_verified','source_count_without_self')}))
if __name__=='__main__':main()
