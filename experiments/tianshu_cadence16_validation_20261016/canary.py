# SPDX-License-Identifier: Apache-2.0
"""Real concurrent canaries and GPU resumed-input checks, before validation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
import time
import urllib.request

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

def request(length, output, input_seed, api_seed=None, temperature=1.0):
    rng = random.Random(input_seed)
    payload = dict(model='minicpm', prompt=[rng.randrange(512, 120000) for _ in range(length)],
                   max_tokens=output, ignore_eos=True, temperature=temperature, top_p=.95,
                   repetition_penalty=1.0, logprobs=None, stream=True,
                   stream_options=dict(include_usage=True))
    if api_seed is not None: payload['seed']=api_seed
    req = urllib.request.Request('http://127.0.0.1:9031/v1/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
    started=time.perf_counter(); text=''; ttft=None; usage=None; text_event_offsets=[]
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            line=raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]': continue
            event=json.loads(line[6:]); choices=event.get('choices', [])
            if choices:
                chunk=choices[0].get('text',''); text+=chunk
                if chunk:
                    offset=time.perf_counter()-started;text_event_offsets.append(offset)
                    if ttft is None:ttft=offset
            if event.get('usage'): usage=event['usage']
    finished=time.perf_counter();duration=finished-started
    assert usage and usage['prompt_tokens']==length and usage['completion_tokens']==output
    assert usage['total_tokens']==length+output and ttft is not None
    return dict(started_perf_counter=started,finished_perf_counter=finished,input_seed=input_seed, api_seed=api_seed, usage=usage, text=text,
                ttft_seconds=ttft, wall_seconds=duration, input_tokens=length, output_tokens=output,
                temperature=temperature,
                text_event_offsets_seconds=text_event_offsets,
                max_text_event_gap_seconds=max((b-a for a,b in zip(text_event_offsets,text_event_offsets[1:])),default=0.),
                mean_decode_token_interval_seconds=(duration-ttft)/(output-1) if output>1 else None)


def canary(root,state,save):
    state['seeded_canaries']=[]
    state['phase']='seeded_canaries';save()
    for _ in range(2):state['seeded_canaries'].append(request(128,16,2026101699,777))
    assert state['seeded_canaries'][0]['text']==state['seeded_canaries'][1]['text']
    actual=[read(p) for p in (root/'prefill_cadence_records').glob('runtime_*.json')]
    used=[r for r in actual if r['calls']>0]
    assert len(used)==1 and used[0]['supported']>0, 'Scheduler guard did not match; inspect recorded configuration'
    state['phase']='concurrent_greedy_canaries';state['concurrent_canaries']=[];save()
    input_arm=root/'verify_resumed_inputs';input_arm.write_text('canary only\n')
    for repeat in range(2):
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(request,4096,16,2026101670+i,777,0.0) for i in range(8)]
            state['concurrent_canaries'].append([f.result() for f in futures])
        save()
    input_arm.unlink()
    input_files=list((root/'prefill_cadence_records').glob('input_checks_*.jsonl'))
    checked=[c for p in input_files for line in p.read_text().splitlines() for c in json.loads(line)['checks']]
    assert all(c['passed'] for c in checked)
    input_hashes={p.name:digest(p) for p in input_files}
    steps=[json.loads(line) for p in (root/'prefill_cadence_records').glob('steps_*.jsonl') for line in p.read_text().splitlines()]
    if state['mode']=='candidate':
        assert any(r['held']>0 for r in steps), 'Concurrent canary failed to exercise deferral'
        assert any(r['history_payload_entries']>0 for r in steps), 'MRV1 token history recovery not observed'
        assert checked, 'No actual resumed GPU input was checked'
    equal_count=sum(a['text']==b['text'] for a,b in zip(*state['concurrent_canaries']))
    gate=dict(complete=True,actual_supported=True,greedy_repeat_equal_count=equal_count,
              greedy_text_equality_not_a_gate=True,actual_resumed_GPU_input_checks=len(checked),
              input_check_journal_hashes=input_hashes,canary_arm_removed_before_throughput=True,
              steps=len(steps),held_steps=sum(r['held']>0 for r in steps),
              history_payload_steps=sum(r['history_payload_entries']>0 for r in steps))
    (root/'canary_route_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
