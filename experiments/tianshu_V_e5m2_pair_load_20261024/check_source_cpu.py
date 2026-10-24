# SPDX-License-Identifier: Apache-2.0
"""Pure-stdlib source-scope prerequisite for the new paired loader."""
import argparse,hashlib,json,sys
from pathlib import Path
import paired_source
import codec
from launch_sweep import CONFIGS
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
source,proof=paired_source.render_reader(codec.READER.read_bytes())
compile(source,'paired_generated.py','exec')
(a.output/'paired_generated.py').write_text(source)
assert [c['loader'] for c in CONFIGS]==['scalar','pair','pair']
assert [c['TILE_SIZE'] for c in CONFIGS]==[32,16,32] and all(c['segments']==4 and c['num_warps']==1 for c in CONFIGS)
assert 'torch' not in sys.modules and 'triton' not in sys.modules
base=Path(__file__).resolve().parent
paths=[Path(__file__).resolve(),Path(paired_source.__file__).resolve(),base/'launch_sweep.py',Path(codec.__file__).resolve(),
       codec.READER,codec.BASE/'CPU_GATE_32g.json',Path('/usr/local/lib/python3.12/site-packages/triton/language/standard.py')]
gate=dict(complete=True,CPU_only=True,source_scope=proof,declared_grid=CONFIGS,
    no_Torch_Triton_import_or_GPU_access=True,word_order_GPU_test_required=True,
    GPU_qualification=False,model_or_service_qualified=False,adopted=False,
    sources={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths})
(a.output/'PAIRED_SOURCE_CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({'complete':True,'sources':len(paths),'only_V_load_and_assignment_changed':True}))
