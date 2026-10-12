# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark anchor layout and sequential Markov proposals on vLLM0.24 transport."""
import json
import os
from pathlib import Path
import torch
from vllm.v1.spec_decode.dflash import DFlashProposer as OriginalDFlashProposer


class DSparkProposer(OriginalDFlashProposer):
    def __init__(self,vllm_config,device,runner=None):
        super().__init__(vllm_config,device,runner)
        assert self.num_speculative_tokens==7
        assert self.parallel_drafting and self.eagle3_use_aux_hidden_state
        assert not self.use_fp64_gumbel, 'The observed CoreX FP64 cast path is unsuitable'
        self.max_query_tokens=self.max_batch_size*self.num_speculative_tokens
        self._recorded_sampling=set()

    def set_inputs_first_pass(self,target_token_ids,next_token_ids,target_positions,
                              target_hidden_states,token_indices_to_sample,cad,
                              num_rejected_tokens_gpu):
        # The old DFlash prepare kernel emits anchor + N masks. DSpark emits
        # N queries (anchor + N-1 masks), sampling all of them, next-token style.
        original_count=self.num_speculative_tokens
        self.num_speculative_tokens=original_count-1
        try:
            total,_,metadata=super().set_inputs_first_pass(
                target_token_ids,next_token_ids,target_positions,target_hidden_states,
                token_indices_to_sample,cad,num_rejected_tokens_gpu)
        finally:
            self.num_speculative_tokens=original_count
        assert total==cad.batch_size()*original_count
        indices=self.arange[:total]
        return total,indices,metadata

    def _sample_draft_tokens(self,hidden_states,sampling_metadata):
        count=self.num_speculative_tokens
        assert hidden_states.shape[0]%count==0
        batch=hidden_states.shape[0]//count
        assert sampling_metadata.temperature.shape[0]==batch
        base=self.model.compute_logits(hidden_states).view(batch,count,-1)
        # Query offset0 is the target's sampled bonus token. Each Markov head
        # conditions the next proposal on the previous sampled draft token.
        anchor_indices=self.arange[:batch].long()*count
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
                'batch':batch,'block_size':count,'query_slots_per_request':count,
                'anchor_is_first_prediction':True,'markov_head_used':True,
                'probabilistic':probs is not None,'target_rejection_sampler_modified':False,
                'confidence_adaptive_verification_enabled':False},indent=2)+'\n')
        return proposals,probs
