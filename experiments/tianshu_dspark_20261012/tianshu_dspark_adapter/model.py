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


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size, rank, prefix):
        super().__init__()
        self.markov_w1=nn.Embedding(vocab_size,rank)
        self.markov_w2=ParallelLMHead(vocab_size,rank,bias=False,disable_tp=True,
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
