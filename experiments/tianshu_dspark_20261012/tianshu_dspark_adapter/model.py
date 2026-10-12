# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/model_executor/models/qwen3_dspark.py.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16, full-vocabulary DSpark draft on the installed DFlash Qwen3 backbone."""
from collections.abc import Iterable
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight


class DSparkOpsProxy:
    """Keep the draft's native APIs while replacing unavailable CUDA _C ops."""
    def __init__(self,original):self.original=original
    def __getattr__(self,name):return getattr(self.original,name)

    def rms_norm(self,out,x,weight,epsilon):
        from flag_gems.ops.rms_norm import rms_norm_out
        return rms_norm_out(out,x,(x.shape[-1],),weight,eps=epsilon)

    def rotary_embedding(self,positions,query,key,head_size,cos_sin_cache,is_neox):
        from flag_gems.fused.rotary_embedding import apply_rotary_pos_emb, apply_rotary_pos_emb_kernel
        import triton
        count=positions.numel()
        q=query.view(count,-1,head_size)
        cosine,sine=cos_sin_cache.chunk(2,dim=-1)
        if key is None:
            # Read input and write a separate output: the observed large NeoX
            # in-place case failed CPU comparison. NUM_K_HEADS=0 removes the
            # unused key loop; separate pointers also avoid read/write overlap.
            rotated=torch.empty_like(q)
            with torch.cuda.device(q.device):
                apply_rotary_pos_emb_kernel[(count,)](
                    rotated,rotated,q,q,cosine,sine,positions,
                    *q.stride(),*q.stride(),*rotated.stride(),*rotated.stride(),
                    positions.stride(0),cosine.stride(0),sine.stride(0),0,
                    q.shape[1],0,head_size,triton.next_power_of_2(head_size),not is_neox,
                    MAX_POSITION_EMBEDDINGS=cosine.shape[0])
            q.copy_(rotated)
            return None
        k=key.view(count,-1,head_size)
        return apply_rotary_pos_emb(q,k,cosine,sine,positions,
                                    rotary_interleaved=not is_neox,inplace=True)


# This changes only the DFlash draft module's reference, never the globally
# shared vllm._custom_ops module or installed sources.
import vllm.model_executor.models.qwen3_dflash as draft_module
draft_module.ops=DSparkOpsProxy(draft_module.ops)


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size, rank, prefix):
        super().__init__()
        self.markov_w1=nn.Embedding(vocab_size,rank)
        self.markov_w2=ParallelLMHead(vocab_size,rank,bias=False,
                                    prefix=maybe_prefix(prefix,'markov_w2'))

    def embed(self, ids):
        return self.markov_w1(ids)


class DSparkConfidenceHead(nn.Module):
    def __init__(self, hidden, rank, prefix):
        super().__init__()
        self.proj=ReplicatedLinear(hidden+rank,1,bias=True,return_bias=False,
                                   params_dtype=torch.float32,prefix=maybe_prefix(prefix,'proj'))

    def forward(self, hidden, markov):
        return self.proj(torch.cat([hidden,markov],dim=-1).float()).squeeze(-1)


class DSparkModel(DFlashQwen3Model):
    def __init__(self,*,vllm_config,start_layer_id=0,prefix=''):
        super().__init__(vllm_config=vllm_config,start_layer_id=start_layer_id,prefix=prefix)
        config=self.config
        assert config.hidden_size==2048 and config.num_hidden_layers==5
        assert config.vocab_size==config.draft_vocab_size==130560
        assert config.markov_head_type=='vanilla' and config.markov_rank==256
        self.markov_head=DSparkMarkovHead(config.vocab_size,config.markov_rank,
                                         maybe_prefix(prefix,'markov_head'))
        self.confidence_head=DSparkConfidenceHead(config.hidden_size,config.markov_rank,
                                                  maybe_prefix(prefix,'confidence_head'))


class DSparkForCausalLM(DFlashQwen3ForCausalLM):
    def __init__(self,*,vllm_config:VllmConfig,prefix=''):
        nn.Module.__init__(self)
        self.config=vllm_config.speculative_config.draft_model_config.hf_config
        assert vllm_config.parallel_config.tensor_parallel_size==1
        assert vllm_config.parallel_config.pipeline_parallel_size==1
        assert vllm_config.model_config.dtype==torch.bfloat16
        assert self.config.vocab_size==vllm_config.model_config.get_vocab_size()
        target_layers=vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        assert target_layers==42
        self.model=DSparkModel(vllm_config=vllm_config,start_layer_id=target_layers,
                               prefix=maybe_prefix(prefix,'model'))
        self.lm_head=ParallelLMHead(self.config.vocab_size,self.config.hidden_size,
                                    prefix=maybe_prefix(prefix,'lm_head'))
        self.logits_processor=LogitsProcessor(self.config.vocab_size,
                                              scale=getattr(self.config,'logit_scale',1.))
        self.draft_id_to_target_id=None
        self.has_own_embed_tokens=False
        self.has_own_lm_head=False

    def markov_embed(self, ids):
        return self.model.markov_head.embed(ids)

    def markov_bias(self, embeddings):
        return self.logits_processor(self.model.markov_head.markov_w2,embeddings)

    def compute_confidence(self, hidden, embeddings):
        return torch.sigmoid(self.model.confidence_head(hidden,embeddings))

    def load_weights(self, weights:Iterable[tuple[str,torch.Tensor]]):
        mapped={}
        for name,value in weights:
            assert 't2d' not in name and 'd2t' not in name
            if 'lm_head' not in name:
                name='model.'+name
            mapped[name]=value
            process_eagle_weight(self,name)
        loader=AutoWeightsLoader(self,skip_substrs=['embed_tokens','lm_head','draft_id_to_target_id'])
        loaded=loader.load_weights(mapped.items())
        # Markov/confidence weights must load even though fixed-depth drafting
        # does not yet use confidence to truncate the verification block.
        assert any('markov_head.markov_w1' in x for x in mapped)
        assert any('markov_head.markov_w2' in x for x in mapped)
        assert any('confidence_head' in x for x in mapped)
        self.model._build_fused_kv_buffers()
        return loaded
