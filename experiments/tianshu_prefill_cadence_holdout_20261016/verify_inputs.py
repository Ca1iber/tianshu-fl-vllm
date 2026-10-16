# SPDX-License-Identifier: Apache-2.0
"""Canary-only exact GPU input check for resumed async requests."""
import functools
import hashlib
import json
from pathlib import Path


def wrap(original, directory, pid):
    directory=Path(directory)
    arm=directory.parent/'verify_resumed_inputs'
    journal=directory/f'input_checks_{pid}.jsonl'

    @functools.wraps(original)
    def prepare(owner, scheduler_output, num_scheduled_tokens):
        result=original(owner,scheduler_output,num_scheduled_tokens)
        if not arm.exists():
            return result
        cached=scheduler_output.scheduled_cached_reqs
        histories=cached.all_token_ids
        output_counts=dict(zip(cached.req_ids,cached.num_output_tokens))
        targets=[]
        for index,req_id in enumerate(owner.input_batch.req_ids):
            if (req_id not in histories or output_counts.get(req_id,0)<=0
                    or scheduler_output.num_scheduled_tokens[req_id]!=1
                    or req_id in owner.input_batch.prev_req_id_to_index):
                continue
            position=int(owner.input_batch.num_computed_tokens_cpu[index])
            assert 0<=position<len(histories[req_id]), 'Resume history lacks required input token'
            targets.append(dict(request_digest=hashlib.sha256(req_id.encode()).hexdigest(),
                input_index=int(owner.query_start_loc.np[index]),expected_position=position,
                history_length=len(histories[req_id]),expected_token=int(histories[req_id][position])))
        if not targets:
            return result
        # This synchronizes only the explicit canary phase. The arm is removed
        # and journal hashes are held constant before measured throughput rounds.
        total=int(sum(num_scheduled_tokens))
        ids=owner.input_ids.gpu[:total].cpu().tolist()
        positions=owner.positions[:total].cpu().tolist()
        for target in targets:
            idx=target['input_index']
            target.update(actual_token=int(ids[idx]),actual_position=int(positions[idx]),
                          one_token_query=True,absent_from_previous_batch=True)
            target['passed']=(target['actual_token']==target['expected_token']
                              and target['actual_position']==target['expected_position'])
        with journal.open('a') as f:
            f.write(json.dumps(dict(mode='canary_only',checks=targets))+'\n')
        assert all(t['passed'] for t in targets), 'Wrong token or position after async resume'
        return result
    return prepare
