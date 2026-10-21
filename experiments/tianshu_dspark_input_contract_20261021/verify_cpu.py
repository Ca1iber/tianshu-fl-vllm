# SPDX-License-Identifier: Apache-2.0
import argparse,json
from pathlib import Path
from common import oracle,sha
p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
r=json.loads((a.raw/'result.json').read_text());assert r['complete'] and r['phase']=='complete' and not r['model_executed'] and len(r['cases'])==160
for path,h in r['frozen_sources'].items():assert sha(path)==h and sha(a.raw/'source_snapshot'/Path(path).relative_to('/'))==h
import torch
torch.set_num_threads(8);values=0
for row in r['cases']:
 assert row['passed'] and row['all_output_and_input_guards_preserved'] and row['RNG_unchanged']
 path=a.raw/(row['name']+'.pt');assert sha(path)==row['tensor_archive_sha256']
 data=torch.load(path,map_location='cpu',weights_only=True);case=data['case']
 expected=oracle(case['context'],data['past'],case['rejected'],data['bonus'].tolist(),data['pages'].tolist())
 assert len(data['actual'])==8
 for key,value in data['actual'].items():assert value.tolist()==expected[key];values+=value.numel()
proof=dict(complete=True,CPU_reverified=True,model_executed=False,official_benchmark=False,cases=len(r['cases']),all_saved_metadata_values_recomputed=values,raw_root=str(a.raw),raw_result_sha256=sha(a.raw/'result.json'),frozen_sources=r['frozen_sources'],all_recorded_guards_and_RNG_passed=True,physical_query_width=7,target_proposal_counts=[1,2,3,7],separate_CPU_limit_policy_tested=True,scope='Actual DSpark host method and original GPU input preparation, no KV writes, model hidden computation, sampling or throughput measured.')
(a.output/'GPU_GATE.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='frozen_sources'}))
