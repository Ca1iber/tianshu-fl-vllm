# SPDX-License-Identifier: Apache-2.0
"""Process-only, source-verified replacement of exactly three metadata writes."""
import ast
import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import textwrap

BASE=Path(__file__).resolve().parents[2]
EXPECTED='b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e'
runpy.run_path(str(BASE/'tianshu_fa2_tiles_20261012/hooks/sitecustomize.py'))


class MetadataLoader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        path=Path(module.__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED:
            raise RuntimeError('Metadata experiment requires the inspected ModelRunner source')
        original=module.ModelRunnerFL._prepare_inputs
        if original.__closure__:
            raise RuntimeError('Cannot replace a metadata function with closure variables')
        source=BASE/'tianshu_metadata_fusion_20261012/experimental_metadata.py'
        spec=importlib.util.spec_from_file_location('tianshu_metadata_probe_kernel',source)
        kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)
        import torch
        from vllm.platforms import current_platform
        recorded=False

        def fill(runner,num_tokens,num_reqs,indices,scheduled):
            nonlocal recorded
            config=runner.vllm_config;model=config.model_config;hf=model.hf_config;parallel=config.parallel_config
            supported=(getattr(current_platform,'vendor_name',None)=='iluvatar'
                and model.dtype==torch.bfloat16 and model.quantization is None
                and config.lora_config is None and config.speculative_config is None
                and not model.is_multimodal_model and not getattr(model,'is_encoder_decoder',False)
                and parallel.tensor_parallel_size==parallel.pipeline_parallel_size==parallel.data_parallel_size==1
                and parallel.decode_context_parallel_size==1
                and getattr(hf,'hidden_size',None)==2048 and getattr(hf,'num_hidden_layers',None)==42
                and getattr(hf,'num_attention_heads',None)==16 and getattr(hf,'num_key_value_heads',None)==2
                and not runner.use_async_spec_decode and not runner.uses_mrope and runner.uses_xdrope_dim==0
                and runner.seq_lens.numel()==runner.num_computed_tokens.numel()
                and 1<=num_tokens<=8192)
            if not supported:
                runner.positions[:num_tokens]=runner.num_computed_tokens[indices].to(torch.int64)+runner.query_pos.gpu[:num_tokens]
                runner.seq_lens[:num_reqs]=runner.num_computed_tokens[:num_reqs]+scheduled
                runner.seq_lens[num_reqs:].fill_(0)
                return
            kernel.fill(runner.num_computed_tokens,indices,runner.query_pos.gpu[:num_tokens],scheduled,
                        runner.positions,runner.seq_lens,num_tokens,num_reqs)
            directory=os.environ.get('TIANSHU_METADATA_RECORD_DIR')
            if directory and not recorded:
                recorded=True;out=Path(directory);out.mkdir(parents=True,exist_ok=True)
                (out/f'metadata_{os.getpid()}.json').write_text(json.dumps({
                    'pid':os.getpid(),'enabled':True,'model_runner_source':str(path),
                    'model_runner_sha256':EXPECTED,'kernel_source':str(source),
                    'kernel_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'num_tokens_first':num_tokens,'num_reqs_first':num_reqs,
                    'max_reqs':runner.num_computed_tokens.numel(),'integer_only':True},indent=2)+'\n')

        pattern=ast.parse('''self.positions[:total_num_scheduled_tokens] = self.num_computed_tokens[req_indices_gpu].to(torch.int64) + self.query_pos.gpu[:total_num_scheduled_tokens]
self.seq_lens[:num_reqs] = self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
self.seq_lens[num_reqs:].fill_(0)
''').body
        wanted=[ast.dump(x,include_attributes=False) for x in pattern]
        tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
        function=tree.body[0];body=function.body;replacements=0;updated=[];index=0
        while index<len(body):
            if index+3<=len(body) and [ast.dump(x,include_attributes=False) for x in body[index:index+3]]==wanted:
                replacement=ast.parse('_tianshu_fill_metadata(self,total_num_scheduled_tokens,num_reqs,req_indices_gpu,num_scheduled_tokens_gpu)').body[0]
                updated.append(ast.copy_location(replacement,body[index]));index+=3;replacements+=1
            else:updated.append(body[index]);index+=1
        if replacements!=1:raise RuntimeError(f'Expected one exact metadata block, found{replacements}')
        function.body=updated;function.name='_tianshu_prepare_inputs';function.decorator_list=[]
        ast.fix_missing_locations(tree)
        namespace=dict(original.__globals__);namespace['_tianshu_fill_metadata']=fill
        exec(compile(tree,str(path)+':metadata_probe','exec'),namespace)
        module.ModelRunnerFL._prepare_inputs=functools.wraps(original)(namespace['_tianshu_prepare_inputs'])


class MetadataFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm_fl.worker.model_runner':return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None or spec.loader is None:raise RuntimeError('ModelRunner unavailable')
        spec.loader=MetadataLoader(spec.loader);return spec


if os.environ.get('TIANSHU_METADATA_FUSION_PROBE')=='1':sys.meta_path.insert(0,MetadataFinder())
