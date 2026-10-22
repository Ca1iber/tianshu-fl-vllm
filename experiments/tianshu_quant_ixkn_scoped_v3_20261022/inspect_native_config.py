import hashlib,json,os,signal,subprocess
from pathlib import Path
import numpy as np
signal.alarm(300)
BASE=Path(__file__).resolve().parent
assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
import torch,vllm_fl,flag_gems
from safetensors import safe_open
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
raw=Path('/workspace/benchmark-results/20261022_ixkn_v3_down_M64_W8_channel_BF16_tianshu');state=json.loads((raw/'app/result.json').read_text());assert state['complete'];source=Path(state['actual_BF16_callable']['source']);before=hashlib.sha256(source.read_bytes()).hexdigest()
with np.load(Path(state['raw_reference'])/state['reference_artifact'],allow_pickle=False) as f:words=f['A_words'].copy();rows=f['rows'].copy();cols=f['columns'].copy();expected=f['native_selected'].copy()
with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',framework='pt',device='cpu') as f:w=f.get_tensor('model.layers.0.mlp.down_proj.weight')
a=torch.from_numpy(words).view(torch.bfloat16).cuda();w=(w*(-.5)).cuda();out=torch.empty((64,2048),device='cuda',dtype=torch.bfloat16)
flag_gems.mm_out(a,w.T,out=out);torch.cuda.synchronize();actual=out[torch.from_numpy(rows).cuda()[:,None],torch.from_numpy(cols).cuda()[None,:]].float().cpu().numpy();assert np.all(abs(actual-expected)<=.02+.01*abs(expected))
entry=flag_gems.mm_out.__globals__['mm_kernel'];chain=[];configs=[]
for depth in range(8):
 chain.append(dict(depth=depth,type=type(entry).__name__))
 if hasattr(entry,'best_config'):
  config=entry.best_config;configs.append(dict(depth=depth,kwargs=config.all_kwargs()))
 entry=getattr(entry,'fn',None)
 if entry is None:break
assert configs and hashlib.sha256(source.read_bytes()).hexdigest()==before
result=dict(complete=True,runner_pid=os.getpid(),original_checkpoint_read_only=True,source=str(source),source_sha256=before,fixture=state['reference_artifact'],selected_output_matches=True,configs=configs,wrapper_chain=chain,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
path=BASE/'NATIVE_CONFIG_32g.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
