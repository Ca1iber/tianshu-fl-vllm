# SPDX-License-Identifier: Apache-2.0
"""Untimed joined eight-prompt request exercises the actual C8 sampler path."""
import json,random,time,urllib.request

def run():
    prompts=[]
    for i in range(8):
        rng=random.Random(202610215000+i);prompts.append([rng.randrange(512,120000) for _ in range(128)])
    body=dict(model='minicpm',prompt=prompts,max_tokens=16,ignore_eos=True,temperature=1.,top_p=.95,seed=777,stream=False,cache_salt='untimed-topp-grid-C8')
    req=urllib.request.Request('http://127.0.0.1:9031/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic()
    with urllib.request.urlopen(req,timeout=900) as response:data=json.load(response)
    usage=data['usage'];assert usage['prompt_tokens']==1024 and usage['completion_tokens']==128 and usage['total_tokens']==1152
    assert len(data['choices'])==8 and all(isinstance(row['text'],str) for row in data['choices'])
    return dict(untimed=True,requests=8,prompt_tokens_per_request=128,completion_tokens_per_request=16,seed=777,response=data,seconds=time.monotonic()-start)
