# SPDX-License-Identifier: Apache-2.0
"""Run the actual current payload builder against a no-network capture session."""
import asyncio,hashlib,inspect,json,os
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',VLLM_PLUGINS='',PYTHONPATH='')
import torch
assert not torch.cuda.is_initialized()
from vllm.benchmarks.lib import endpoint_request_func as endpoint
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
BASE=Path(__file__).resolve().parent
class CaptureSession:
 def __init__(self):self.payloads=[]
 def post(self,**kwargs):
  self.payloads.append(dict(kwargs['json']))
  raise RuntimeError('CPU_PAYLOAD_CAPTURE_NO_REQUEST_SENT')
session=CaptureSession()
request=endpoint.RequestFuncInput(prompt='CPU payload audit only',api_url='http://127.0.0.1:9031/v1/completions',prompt_len=4096,output_len=1024,model='minicpm',ignore_eos=True,extra_body={})
assert endpoint.ASYNC_REQUEST_FUNCS['vllm'] is endpoint.async_request_openai_completions
result=asyncio.run(endpoint.ASYNC_REQUEST_FUNCS['vllm'](request,session))
assert len(session.payloads)==1 and 'CPU_PAYLOAD_CAPTURE_NO_REQUEST_SENT' in result.error
payload=session.payloads[0];assert 'temperature' not in payload and 'top_p' not in payload and payload['ignore_eos'] is True and payload['max_tokens']==1024 and payload['repetition_penalty']==1
model_gen=json.loads(Path('/workspace/MiniCPM5-2B/generation_config.json').read_text());params=CompletionRequest(**payload).to_sampling_params(max_tokens=1024,default_sampling_params={k:model_gen[k] for k in ('temperature','top_p')})
assert params.temperature==1.0 and params.top_p==.95 and not torch.cuda.is_initialized()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
logs={}
for name in ('20261021_m_range_prefill_resume_official4k_tianshu','20261021_nn_silu_fx_official4k_tianshu'):
 path=Path('/workspace/benchmark-results')/name/'benchmark.log';text=path.read_text();count=text.count('vllm bench serve no longer sets temperature==0');assert count==4
 logs[name]=dict(sha256=sha(path),current_CLI_default_warning_count=count)
files=[Path(inspect.getsourcefile(endpoint)),Path('/usr/local/lib/python3.12/site-packages/vllm/benchmarks/serve.py'),Path('/usr/local/lib/python3.12/site-packages/vllm/entrypoints/cli/benchmark/serve.py'),Path(inspect.getsourcefile(CompletionRequest)),Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py'),Path('/workspace/MiniCPM5-2B/generation_config.json')]
gate=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,HTTP_requests_sent=0,actual_payload_builder_executed=True,backend='vllm',actual_provider='vllm.benchmarks.lib.endpoint_request_func.async_request_openai_completions',constructed_payload=payload,native_protocol_effective_parameters=dict(temperature=params.temperature,top_p=params.top_p,repetition_penalty=params.repetition_penalty),protocol_merge_scope='Native CompletionRequest with unmodified local model generation defaults; not a live Sampler tracing claim.',old_and_new_formal_logs=logs,source_hashes={str(p):sha(p) for p in files},verifier_sha256=sha(__file__),legacy_backend_file_not_current_CLI_provider=True,benchmark_not_modified=True)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('source_hashes','old_and_new_formal_logs')}))
