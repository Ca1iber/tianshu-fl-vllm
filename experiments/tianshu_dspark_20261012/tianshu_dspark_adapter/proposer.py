# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark anchor layout and sequential Markov proposals on vLLM0.24 transport."""
import json
import os
from pathlib import Path
import torch
from vllm.v1.spec_decode.dflash import DFlashProposer as OriginalDFlashProposer
from vllm.forward_context import set_forward_context


class DSparkProposer(OriginalDFlashProposer):
    def __init__(self,vllm_config,device,runner=None):
        super().__init__(vllm_config,device,runner)
        assert self.num_speculative_tokens in (1,2,3,7)
        self.backbone_block_size=7
        assert self.parallel_drafting and self.eagle3_use_aux_hidden_state
        assert not self.use_fp64_gumbel, 'The observed CoreX FP64 cast path is unsuitable'
        self.max_query_tokens=self.max_batch_size*self.backbone_block_size
        self.max_positions=max(self.max_positions,self.max_num_tokens+self.max_query_tokens)
        if self.positions.numel()<self.max_query_tokens:
            self.positions=torch.zeros(self.max_query_tokens,dtype=torch.int64,device=device)
        if self._slot_mapping_buffer.numel()<self.max_query_tokens:
            self._slot_mapping_buffer=torch.zeros(self.max_query_tokens,dtype=torch.int64,device=device)
        if self.arange.numel()<self.max_positions+1:
            self.arange=torch.arange(self.max_positions+1,device=device,dtype=torch.int32)
        root=Path(os.environ['TIANSHU_DSPARK_RECORD_DIR'])/'query_buffers'
        root.mkdir(parents=True,exist_ok=True)
        (root/f'capacity_{os.getpid()}.json').write_text(json.dumps({
            'max_batch_size':self.max_batch_size,'backbone_query_slots':self.backbone_block_size,
            'max_query_tokens':self.max_query_tokens,'positions_capacity':self.positions.numel(),
            'slot_mapping_capacity':self._slot_mapping_buffer.numel(),'arange_capacity':self.arange.numel(),
            'max_context_tokens':self.max_num_tokens,'target_proposals':self.num_speculative_tokens},indent=2)+'\n')
        self._recorded_sampling=set()

    @torch.inference_mode()
    def dummy_run(self,num_tokens,use_cudagraphs=True,is_graph_capturing=False,slot_mappings=None):
        if num_tokens<=0 or (os.getenv('TIANSHU_DSPARK_MATCHED_WARMUP')!='1' and self.num_speculative_tokens==7):
            return super().dummy_run(num_tokens,use_cudagraphs,is_graph_capturing,slot_mappings)
        # Target uniform verification has K+1=8 rows per request, while
        # DSpark has K=7 queries including its anchor. Warm the real drafter
        # query sizes; keep the original unpadded context size and KV API.
        target_width=self.num_speculative_tokens+1
        batch=min(self.max_batch_size,max(1,(num_tokens+target_width-1)//target_width))
        queries=batch*7
        mode,padded,across_dp=self._determine_batch_execution_and_padding(
            queries,use_cudagraphs=use_cudagraphs)
        if (self._draft_attn_layer_names and slot_mappings is not None and
            next(iter(self._draft_attn_layer_names)) in slot_mappings):
            slots=self._get_slot_mapping(padded)
        else:
            slots=slot_mappings or {}
        context=self.hidden_states[:num_tokens]
        positions=self._context_positions_buffer[:num_tokens]
        self.model.precompute_and_store_context_kv(context,positions)
        with set_forward_context(None,self.vllm_config,num_tokens=padded,
                num_tokens_across_dp=across_dp,cudagraph_runtime_mode=mode,slot_mapping=slots):
            output=self.model(input_ids=self.input_ids[:padded],
                              positions=self._get_positions(padded),inputs_embeds=None)
        assert isinstance(output,torch.Tensor)
        # These heads run outside the target graph and are skipped by the
        # parent dummy run. No token sampling or RNG draw is performed.
        sampled_hidden=output[:queries].reshape(batch,7,-1)[:,:self.num_speculative_tokens].reshape(
            batch*self.num_speculative_tokens,-1)
        self.model.compute_logits(sampled_hidden)
        for dtype in {self.input_ids.dtype,torch.int64}:
            dummy_ids=torch.zeros(batch,dtype=dtype,device=self.device)
            self.model.markov_bias(self.model.markov_embed(dummy_ids))
        root=Path(os.environ['TIANSHU_DSPARK_RECORD_DIR'])/'matched_warmup'
        root.mkdir(parents=True,exist_ok=True)
        (root/f'context_{num_tokens}_queries_{queries}.json').write_text(json.dumps({
            'target_context_tokens':num_tokens,'draft_queries':queries,'draft_batch':batch,
                'padded_draft_queries':padded,'sampled_positions':self.num_speculative_tokens,
                'lm_head_warmed':True,'markov_head_warmed':True,
            'rng_draws_added':False,'target_rejection_sampler_modified':False},indent=2)+'\n')

    def set_inputs_first_pass(self,target_token_ids,next_token_ids,target_positions,
                              target_hidden_states,token_indices_to_sample,cad,
                              num_rejected_tokens_gpu):
        # The old DFlash prepare kernel emits anchor + N masks. DSpark emits
        # N queries (anchor + N-1 masks), sampling all of them, next-token style.
        original_count=self.num_speculative_tokens
        self.num_speculative_tokens=self.backbone_block_size-1
        try:
            total,_,metadata=super().set_inputs_first_pass(
                target_token_ids,next_token_ids,target_positions,target_hidden_states,
                token_indices_to_sample,cad,num_rejected_tokens_gpu)
        finally:
            self.num_speculative_tokens=original_count
        assert total==cad.batch_size()*self.backbone_block_size
        # In this anchor-first parallel lane all query rows are sampled, in
        # order. BaseProposer indexes once then immediately exits through its
        # parallel branch. A prefix view avoids an otherwise redundant gather.
        indices=(slice(0,total) if os.getenv('TIANSHU_DSPARK_SAMPLE_VIEW')=='1'
                 else self.arange[:total])
        return total,indices,metadata

    def _sample_draft_tokens(self,hidden_states,sampling_metadata):
        count=self.num_speculative_tokens
        assert hidden_states.shape[0]%self.backbone_block_size==0
        batch=hidden_states.shape[0]//self.backbone_block_size
        if not sampling_metadata.all_greedy:
            assert sampling_metadata.temperature is not None
            assert sampling_metadata.temperature.shape[0]==batch
        # Keep all seven trained noncausal query slots in the backbone, but
        # compute vocabulary logits and sequential transitions only for the
        # requested prefix. The original target sampler still verifies every
        # emitted proposal against its actual draft probability.
        sampled_hidden=hidden_states.reshape(batch,self.backbone_block_size,-1)[:,:count].reshape(
            batch*count,-1)
        base=self.model.compute_logits(sampled_hidden).view(batch,count,-1)
        # Query offset0 is the target's sampled bonus token. Each Markov head
        # conditions the next proposal on the previous sampled draft token.
        anchor_indices=self.arange[:batch].long()*self.backbone_block_size
        previous=self.input_ids[anchor_indices]
        ids,probabilities=[],[]
        for position in range(count):
            embedding=self.model.markov_embed(previous)
            logits=base[:,position]+self.model.markov_bias(embedding)
            sampled,probs=self._sample_from_logits(logits,sampling_metadata)
            ids.append(sampled)
            if probs is not None:
                probabilities.append(probs)
            previous=sampled
        proposals=torch.stack(ids,dim=1).reshape(-1)
        probs=(torch.stack(probabilities,dim=1).reshape(batch*count,-1)
               if probabilities else None)
        record_key=(batch,probs is not None)
        if record_key not in self._recorded_sampling:
            self._recorded_sampling.add(record_key)
            root=Path(os.environ['TIANSHU_DSPARK_RECORD_DIR']);root.mkdir(parents=True,exist_ok=True)
            (root/f'sequential_sampler_{os.getpid()}_{batch}_{int(probs is not None)}.json').write_text(json.dumps({
                'batch':batch,'block_size':count,'query_slots_per_request':self.backbone_block_size,
                'backbone_block_size':self.backbone_block_size,'sampled_prefix_positions':count,
                'anchor_is_first_prediction':True,'markov_head_used':True,
                'probabilistic':probs is not None,
                'target_rejection_sampler_modified':os.getenv('TIANSHU_DSPARK_FP32_THRESHOLDS')=='1',
                'target_rejection_kernel_modified':False,
                'positive_fp32_rejection_thresholds':os.getenv('TIANSHU_DSPARK_FP32_THRESHOLDS')=='1',
                'confidence_adaptive_verification_enabled':False},indent=2)+'\n')
        return proposals,probs
