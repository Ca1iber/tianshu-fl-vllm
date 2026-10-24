# SPDX-License-Identifier: Apache-2.0
"""Independent synthetic CPU projections split by actual Q/K/V output rows."""
import csv,importlib.util,json,statistics,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
ROOT=Path('/dev/shm/tianshu_W4_QKV_affine_fit_CPU_20261024')
spec=importlib.util.spec_from_file_location('_QKV_parts_original',BASE.parent/'tianshu_W4_attention64_20261024/common.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
r=json.loads((ROOT/'result.json').read_text());proof=json.loads((BASE/'INDEPENDENT_CPU_GATE_32g.json').read_text())
assert proof['complete'] and c.sha(ROOT/'result.json')==proof['raw_result_sha256']
rows=[]
for layer in (0,20,41):
    words=c.original(layer,'QKV');actual=c.f32(words).astype(np.float64).T
    iq,isc,iz,_=c.quant(words,'affine')
    data=np.load(ROOT/f'QKV_L{layer}_fitted.npz',allow_pickle=False)
    weights={'minmax':c.f32(c.dequant(iq,isc,iz)).astype(np.float64),
             'fitted':c.f32(c.dequant(c.unpack(data['w']),data['s'],data['z'])).astype(np.float64)}
    for phase in range(3):
        x=c.f32(c.inputs(layer,'QKV',phase)[:8]).astype(np.float64);reference=x@actual
        for mode,w in weights.items():
            output=x@w
            for part,start,end in [('Q',0,2048),('K',2048,2304),('V',2304,2560)]:
                target=reference[:,start:end];value=output[:,start:end]
                rows.append(dict(layer=layer,phase=phase,variant=mode,part=part,
                    full_FP64_cells=int(target.size),relative_L2=float(np.linalg.norm(value-target)/np.linalg.norm(target))))
summary={part:{mode:statistics.mean(v['relative_L2'] for v in rows if v['part']==part and v['variant']==mode)
    for mode in weights} for part in ('Q','K','V')}
with (BASE/'KEY_PART_ERRORS_32g.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
result=dict(complete=True,CPU_only=True,only3_layers_3phases_8rows=True,summary=summary,
    raw_result_sha256=c.sha(ROOT/'result.json'),source_sha256=c.sha(__file__),reader_source_sha256=c.sha(Path(c.__file__)),
    csv_sha256=c.sha(BASE/'KEY_PART_ERRORS_32g.csv'),GPU_qualified=False,model_quality_qualified=False)
(BASE/'PARTS_CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
