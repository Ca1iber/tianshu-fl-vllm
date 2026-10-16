# SPDX-License-Identifier: Apache-2.0
"""CPU-only wrapper contract; this is not a full Scheduler or service test."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import policy
import verify_inputs
import tempfile


def check_input_gate():
    class FakeGPU:
        def __init__(self, values):self.values=values
        def __getitem__(self, key):return FakeGPU(self.values[key])
        def cpu(self):return self
        def tolist(self):return self.values
    with tempfile.TemporaryDirectory(prefix='tianshu_resume_gate_') as temporary:
        root=Path(temporary);directory=root/'records';directory.mkdir()
        arm=root/'verify_resumed_inputs'
        output=object();original=lambda *args:output
        request=NS(scheduled_cached_reqs=NS(all_token_ids={'r':[0]*4097+[42]},
                     req_ids=['r'],num_output_tokens=[2]),num_scheduled_tokens={'r':1})
        owner=NS(input_batch=NS(req_ids=['r'],prev_req_id_to_index={},num_computed_tokens_cpu=[4097]),
                 query_start_loc=NS(np=[0,1]),input_ids=NS(gpu=FakeGPU([42])),positions=FakeGPU([4097]))
        check=verify_inputs.wrap(original,directory,1)
        # No arm: neither GPU input nor output journal is touched.
        owner.input_ids.gpu=None;assert check(owner,request,[1]) is output
        assert not list(directory.iterdir())
        owner.input_ids.gpu=FakeGPU([42]);arm.write_text('CPU fixture')
        assert check(owner,request,[1]) is output
        journal=directory/'input_checks_1.jsonl';before=journal.read_bytes()
        record=json.loads(journal.read_text());assert record['checks'][0]['passed']
        for values,position in (([43],4097),([42],4096)):
            owner.input_ids.gpu=FakeGPU(values);owner.positions=FakeGPU([position])
            try:check(owner,request,[1])
            except AssertionError:pass
            else:raise AssertionError('input verification accepted wrong GPU value')
        owner.input_ids.gpu=FakeGPU([42]);owner.positions=FakeGPU([4097])
        request.scheduled_cached_reqs.all_token_ids={'r':[0]*4097}
        try:check(owner,request,[1])
        except AssertionError:pass
        else:raise AssertionError('missing history accepted')
        request.scheduled_cached_reqs.all_token_ids={'r':[0]*4097+[42]}
        owner.input_batch.prev_req_id_to_index={'r':0}
        stable=journal.read_bytes();check(owner,request,[1]);assert journal.read_bytes()==stable
        arm.unlink();owner.input_ids.gpu=None
        assert check(owner,request,[1]) is output and journal.read_bytes()==stable
    return True


def owner():
    request = NS(request_id='decode', num_tokens_with_spec=4097, num_output_placeholders=0,
                 num_computed_tokens=4096, num_prompt_tokens=4096, next_decode_eligible_step=0)
    waiting = NS(status=NS(name='WAITING'), num_tokens=4096, num_computed_tokens=0)
    parallel = NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                  decode_context_parallel_size=1, prefill_context_parallel_size=1,
                  worker_cls='vllm_fl.worker.worker.WorkerFL')
    model = NS(is_multimodal_model=False, hf_config=NS(hidden_size=2048, num_hidden_layers=42,
                                                    num_attention_heads=16, num_key_value_heads=2))
    return NS(vllm_config=NS(model_config=model, speculative_config=None), parallel_config=parallel,
              use_v2_model_runner=False, has_mamba_layers=False, is_encoder_decoder=False,
              lora_config=None, connector=None, ec_connector=None,
              scheduler_config=NS(enable_chunked_prefill=True, async_scheduling=True),
              policy=NS(name='FCFS'), max_model_len=131072, block_size=16,
              current_step=0, max_num_running_reqs=256, max_num_scheduled_tokens=2048,
              running=[request], waiting=[waiting],
              kv_cache_manager=NS(block_pool=NS(get_num_free_blocks=lambda:10000)))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scheduler-source', type=Path, required=True)
    args = parser.parse_args()
    raw = args.scheduler_source.read_bytes()
    expected = 'df78076164fe02a63876d7690cefac810f1ba4d8fb58afc15e786c1608064090'
    assert hashlib.sha256(raw).hexdigest() == expected
    module = ast.parse(raw)
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    assert [a.arg for a in method.args.args] == ['self', 'throttle_prefills']
    assert len(method.args.defaults) == 1 and method.args.defaults[0].value is False
    branch = next(n for n in ast.walk(method) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'self.current_step < request.next_decode_eligible_step')
    assert isinstance(branch.body[-1], ast.Continue)
    # A small schedule stub implements only the exact eligibility contract above.
    calls, observations = [], []
    def original(s, throttle_prefills=False):
        calls.append(throttle_prefills); s.current_step += 1
        return NS(num_scheduled_tokens={r.request_id:1 for r in s.running
                                       if s.current_step >= r.next_decode_eligible_step})
    s = owner(); req = s.running[0]
    wrapped = policy.wrap(original, enabled=True, observe=lambda *v:observations.append(v[-1]))
    for _ in range(12):
        wrapped(s)
        assert req.next_decode_eligible_step == 0
    assert [o['held'] for o in observations] == [1,1,0]*4
    assert len(calls) == 12 and all(v is False for v in calls)
    assert all(o['scheduled_held_requests'] == 0 for o in observations)
    # Default / positional / keyword forwarding and output object identity.
    token = NS(num_scheduled_tokens={})
    forwarded = []
    def identity(s, throttle_prefills=False):
        forwarded.append(throttle_prefills); s.current_step += 1; return token
    f = policy.wrap(identity, enabled=False, observe=lambda *v:None)
    s = owner(); assert f(s) is token and f(s, True) is token and f(s, throttle_prefills=True) is token
    assert forwarded == [False, True, True]
    # Fallbacks must not change any eligibility, even on the first cadence step.
    def fallback(mutate, throttle=False):
        s = owner(); mutate(s); seen = []
        f = policy.wrap(original, enabled=True, observe=lambda *v:seen.append(v[-1]))
        f(s, throttle_prefills=throttle)
        assert s.running[0].next_decode_eligible_step == 0 and not seen[-1]['defer']
    fallback(lambda s:s.waiting.clear())
    fallback(lambda s:setattr(s, 'has_mamba_layers', True))
    fallback(lambda s:setattr(s, 'connector', object()))
    fallback(lambda s:setattr(s.vllm_config, 'speculative_config', object()))
    fallback(lambda s:setattr(s.parallel_config, 'tensor_parallel_size', 2))
    fallback(lambda s:setattr(s.scheduler_config, 'async_scheduling', False))
    fallback(lambda s:setattr(s.kv_cache_manager.block_pool, 'get_num_free_blocks', lambda:511))
    fallback(lambda s:None, throttle=True)
    # Running partial prefill alone is pressure; only decode requests are held.
    s = owner(); s.waiting.clear()
    partial = NS(request_id='prefill', num_computed_tokens=1, num_prompt_tokens=4096,
                 num_tokens_with_spec=4096, num_output_placeholders=0, next_decode_eligible_step=0)
    s.running.append(partial); seen=[]
    f=policy.wrap(original, enabled=True, observe=lambda *v:seen.append(v[-1]))
    output=f(s); assert seen[-1]['held']==1 and 'prefill' in output.num_scheduled_tokens
    # Exception restores eligibility even if original removes a request.
    s=owner(); req=s.running[0]
    def fail(s, throttle_prefills=False):
        s.running.clear(); raise ValueError('controlled')
    try:policy.wrap(fail, enabled=True, observe=lambda *v:None)(s)
    except ValueError:pass
    else:raise AssertionError('exception swallowed')
    assert req.next_decode_eligible_step==0
    # Do not lower a pre-existing eligibility or overwrite a legitimate update.
    s=owner(); req=s.running[0];req.next_decode_eligible_step=100
    policy.wrap(original, enabled=True, observe=lambda *v:None)(s)
    assert req.next_decode_eligible_step==100
    def update(s, throttle_prefills=False):
        s.running[0].next_decode_eligible_step=200;return NS(num_scheduled_tokens={})
    s=owner();policy.wrap(update, enabled=True, observe=lambda *v:None)(s)
    assert s.running[0].next_decode_eligible_step==200
    # Both configured flags use the existing MRV1 metadata path in the guarded
    # candidate; control retains its original flag. Restore it on exceptions too.
    for flag in (False,True):
        for enabled in (False,True):
            s=owner();s.use_v2_model_runner=flag;observed=[]
            def metadata_contract(s,throttle_prefills=False):
                observed.append(s.use_v2_model_runner)
                return NS(num_scheduled_tokens={})
            policy.wrap(metadata_contract,enabled=enabled,observe=lambda *v:None)(s)
            assert observed==[False if enabled else flag]
            assert s.use_v2_model_runner==flag
        s=owner();s.use_v2_model_runner=flag
        try:policy.wrap(fail,enabled=True,observe=lambda *v:None)(s)
        except ValueError:pass
        else:raise AssertionError('exception swallowed')
        assert s.use_v2_model_runner==flag
    cached=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_make_cached_request_data')
    assert any(isinstance(n,ast.If) and ast.unparse(n.test)=='not self.use_v2_model_runner' for n in ast.walk(cached))
    worker=Path(__file__).resolve().parents[2]/'vllm_fl/worker/worker.py'
    assert 'self.model_runner = ModelRunnerFL(self.vllm_config, self.device)' in worker.read_text()
    assert check_input_gate()
    result=dict(complete=True, actual_scheduler_signature_and_eligibility_branch_checked=True,
                stub_only_not_full_scheduler=True, single_call_argument_forwarding_identity_checked=True,
                twelve_step_cadence_and_restoration_checked=True, fallback_cases=8,
                exception_removal_existing_eligibility_and_legitimate_update_checked=True,
                both_configured_flags_metadata_selection_and_restore_checked=True,
                actual_cached_data_recovery_branch_and_WorkerFL_constructor_checked=True,
                canary_input_helper_CPU_positive_negative_and_disabled_cases_checked=True,
                resume_input_helper_sha256=hashlib.sha256(Path(verify_inputs.__file__).read_bytes()).hexdigest(),
                producer_model_and_GPU_not_run=True, scheduler_source_sha256=expected,
                policy_sha256=hashlib.sha256(Path(policy.__file__).read_bytes()).hexdigest())
    print(json.dumps(result))


if __name__=='__main__':main()
